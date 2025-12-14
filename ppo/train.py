# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import json
import logging
import os
import pathlib
import pdb  # noqa: F401
import random
import sai_mujoco  # noqa: F401
import time
import torch
import tyro
import wandb

import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.optim as optim

from dataclasses import dataclass
from gymnasium import spaces
from gymnasium.vector import SyncVectorEnv
from torch.distributions.categorical import Categorical
from typing import Optional

from ppo.t1 import LowerT1JoyStick

logger = logging.getLogger(__name__)


def quat_rotate_inverse(q_batch: np.ndarray, v_batch: np.ndarray):
    """
    Rotates vectors `v_batch` by the inverse of quaternions `q_batch`.

    Args:
        q_batch (np.ndarray): Quaternions in the format (..., 4) [x, y, z, w].
        v_batch (np.ndarray): 3D vectors in the format (..., 3).

    Returns:
        np.ndarray: The rotated vectors in the format (..., 3).
    """
    q_w = q_batch[..., -1]
    q_vec = q_batch[..., :3]
    a = v_batch * (2.0 * q_w**2 - 1.0)[..., None]
    b = np.cross(q_vec, v_batch) * (q_w * 2.0)[..., None]
    c = q_vec * (np.sum(q_vec * v_batch, axis=-1, keepdims=True) * 2.0)
    return a - b + c


class CommandActionWrapper(gym.ActionWrapper):
    def __init__(self, env, use_bc_controller=False):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.lower_control = LowerT1JoyStick(self.base_env)
        # RL policy outputs 7 discrete actions: no-op, forward, backward, left, right, rotate left, rotate right
        self.action_space = spaces.Discrete(7)

    @property
    def single_observation_space(self):
        return self.observation_space

    @property
    def single_action_space(self):
        return self.action_space

    def action(self, command):
        # Map discrete action to high-level commands
        commands = [
            [0, 0, 0],  # no-op
            [1.5, 0, 0],  # forward
            [-1.5, 0, 0],  # backward
            [0, 1.5, 0],  # move left
            [0, -1.5, 0],  # move right
            [0, 0, 1.5],  # rotate left
            [0, 0, -1.5],  # rotate right
        ]
        command_float = np.array(commands[command], dtype=np.float32)
        # NOTE: relies on base_env private getters; works but is brittle.
        observation = self.base_env._get_obs()  # type: ignore
        info = self.base_env._get_info()  # type: ignore
        ctrl = self.lower_control.get_actions(command_float, observation, info)
        return ctrl


class SkipStepsWrapper(gym.Wrapper):
    def __init__(self, env, skip_steps=50):
        super().__init__(env)
        self.skip_steps = skip_steps
        self.step_count = 0
        self.current_action = None
        self.start_obs = None
        self.info = {
            "robot_fallen": 0,
            "offside": 0,
            "ball_blocked": 0,
            "goal_scored": 0,
            "ball_vel_twd_goal": 0,
            "robot_distance_ball": 0,
            "robot_quat": [],
        }

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        self.start_obs = obs
        self.info = {k: 0 for k in self.info}
        return obs, info

    def step(self, action):
        if self.step_count % self.skip_steps == 0:
            self.current_action = action
            self.rewards = 0.0
            self.done = False
            self.truncated = False

        # Apply the current_action for skip_steps real steps
        steps_taken = 0
        for _ in range(self.skip_steps):
            next_obs, _, done, truncated, base_info = self.env.step(self.current_action)
            reward, info = get_reward_and_info(base_info.get("reward_terms", {}))

            self.rewards += float(reward)

            try:
                self.info = {key: self.info[key] + info[key] for key in self.info}
            except KeyError:
                pass

            steps_taken += 1
            if done or truncated:
                self.done = done
                self.truncated = truncated
                self.info = info
                break

        self.step_count += steps_taken
        self.info["robot_quat"] = base_info.get("robot_quat", [])

        # Return the transition: next_obs is the obs after the block, reward is sum
        return next_obs, self.rewards, self.done, self.truncated, self.info


def get_reward_and_info(reward_terms):
    robot_fallen = int(reward_terms.get("robot_fallen", 0))
    offside = int(reward_terms.get("offside", 0))
    ball_blocked = int(reward_terms.get("ball_blocked", 0))
    goal_scored = int(reward_terms.get("goal_scored", 0))
    robot_distance_ball = float(reward_terms.get("robot_distance_ball", 0.0))
    ball_vel_twd_goal = float(reward_terms.get("ball_vel_twd_goal", 0.0))

    info = {
        "robot_fallen": robot_fallen,
        "offside": offside,
        "ball_blocked": ball_blocked,
        "goal_scored": goal_scored,
        "robot_distance_ball": robot_distance_ball,
        "ball_vel_twd_goal": ball_vel_twd_goal,
    }

    reward = 0.0
    reward += robot_fallen * -100.0
    reward += offside * -100.0
    reward += ball_blocked * -100.0
    reward += goal_scored * 100.0
    reward += ball_vel_twd_goal * 0.03
    reward += np.clip(robot_distance_ball, -np.inf, 0.4) * 0.005
    reward -= 0.0015  # step penalty to encourage faster solutions

    return reward, info


def evaluate(agent, env_id, device, num_episodes=10, use_bc_controller=False):
    env = make_env(env_id, 0, False, "", use_bc_controller)()

    episode_rewards = []
    episode_infos = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=random.randint(0, 1000000))
        if "robot_quat" in info:
            quat = info["robot_quat"]
            obs = obs.copy()
            obs[24:27] = quat_rotate_inverse(quat, obs[24:27])  # ball pos
            obs[27:30] = quat_rotate_inverse(quat, obs[27:30])  # ball lin vel
            obs[30:33] = quat_rotate_inverse(quat, obs[30:33])  # ball ang vel
            obs[33:36] = quat_rotate_inverse(quat, obs[33:36])  # target pos
            obs[39:42] = quat_rotate_inverse(quat, obs[39:42])  # target lin vel
            obs[42:45] = quat_rotate_inverse(quat, obs[42:45])  # target lin vel
            obs = np.concatenate([obs, quat])
        else:
            raise ValueError("robot_quat not found in info during evaluation reset.")
        done = False
        truncated = False
        episode_reward = 0.0
        episode_info = {
            "robot_fallen": 0,
            "offside": 0,
            "ball_blocked": 0,
            "goal_scored": 0,
            "ball_vel_twd_goal": 0.0,
            "robot_distance_ball": 0.0,
        }
        while not (done or truncated):
            with torch.no_grad():
                action = agent.get_action(torch.Tensor(obs).unsqueeze(0).to(device))
                action = action.cpu().numpy()[0]
            next_obs, reward, done, truncated, info = env.step(action)
            if "robot_quat" in info:
                quat = info["robot_quat"]
                next_obs = next_obs.copy()
                next_obs[24:27] = quat_rotate_inverse(quat, next_obs[24:27])  # type: ignore # ball pos
                next_obs[27:30] = quat_rotate_inverse(quat, next_obs[27:30])  # type: ignore # ball lin vel
                next_obs[30:33] = quat_rotate_inverse(
                    quat, next_obs[30:33]
                )  # ball ang vel #type: ignore
                next_obs[33:36] = quat_rotate_inverse(
                    quat, next_obs[33:36]
                )  # ball ang vel #type: ignore
                next_obs[39:42] = quat_rotate_inverse(
                    quat, next_obs[39:42]
                )  # target lin vel #type: ignore
                next_obs[42:45] = quat_rotate_inverse(
                    quat, next_obs[42:45]
                )  # target ang vel #type: ignore
                next_obs = np.concatenate([next_obs, quat])
            else:
                raise ValueError("robot_quat not found in info during evaluation step.")

            episode_reward += float(reward)
            for key in episode_info:
                episode_info[key] += info.get(key, 0)

            obs = next_obs

        episode_rewards.append(episode_reward)
        episode_infos.append(episode_info)

    env.close()

    rewards_mean = np.mean(episode_rewards)
    rewards_std = np.std(episode_rewards)

    robot_fallen_total = sum(ep["robot_fallen"] for ep in episode_infos)
    offside_total = sum(ep["offside"] for ep in episode_infos)
    ball_blocked_total = sum(ep["ball_blocked"] for ep in episode_infos)
    goal_scored_total = sum(ep["goal_scored"] for ep in episode_infos)
    ball_vel_twd_goal_values = [ep["ball_vel_twd_goal"] for ep in episode_infos]
    ball_vel_twd_goal_mean = np.mean(ball_vel_twd_goal_values)
    ball_vel_twd_goal_std = np.std(ball_vel_twd_goal_values)
    robot_distance_ball_values = [ep["robot_distance_ball"] for ep in episode_infos]
    robot_distance_ball_mean = np.mean(robot_distance_ball_values)
    robot_distance_ball_std = np.std(robot_distance_ball_values)

    return (
        rewards_mean,
        rewards_std,
        robot_fallen_total,
        offside_total,
        ball_blocked_total,
        goal_scored_total,
        ball_vel_twd_goal_mean,
        ball_vel_twd_goal_std,
        robot_distance_ball_mean,
        robot_distance_ball_std,
    )


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None  # type: ignore
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "LowerT1ObstaclePenaltyKick-v0"
    """the id of the environment"""
    total_timesteps: int = 1000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None  # type: ignore
    """the target KL divergence threshold"""
    eval_interval: int = 20
    """iterations between evaluations"""
    checkpoint_interval: int = 100
    """iterations between checkpoints"""
    resume_from: Optional[str] = None
    """path to checkpoint directory to resume from"""
    weight_decay: float = 0.0
    """weight decay for the optimizer"""
    use_bc_controller: bool = False
    """if toggled, use behavior cloning controller instead of RL-trained controller"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, use_bc_controller=False):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        # env = gym.wrappers.RecordEpisodeStatistics(env)
        env = CommandActionWrapper(env, use_bc_controller=use_bc_controller)
        env = SkipStepsWrapper(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_prod = int(np.array(envs.single_observation_space.shape).prod())
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_prod, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_prod, 256)),
            nn.ELU(),
            layer_init(nn.Linear(256, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 128)),
            nn.ELU(),
            layer_init(nn.Linear(128, 7), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        logits = torch.clamp(logits, -10, 10)  # Prevent NaN in Categorical
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()
        else:
            action = action.long()
        logprob = probs.log_prob(action)
        entropy = probs.entropy()
        return action, logprob, entropy, self.critic(x)

    def get_action(self, x):
        logits = self.actor(x)
        logits = torch.clamp(logits, -10, 10)  # Prevent NaN in Categorical
        action = torch.argmax(logits, dim=-1)
        return action


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    checkpoint_dir = f"checkpoints/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if args.track:
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = SyncVectorEnv(
        [
            make_env(
                args.env_id, i, args.capture_video, run_name, args.use_bc_controller
            )
            for i in range(args.num_envs)
        ],
    )
    # Update observation space to include quaternion
    if isinstance(envs.single_observation_space, spaces.Box):
        original_shape = envs.single_observation_space.shape
        new_shape = (original_shape[0] + 4,)
        low = np.concatenate([envs.single_observation_space.low, -np.ones(4)])
        high = np.concatenate([envs.single_observation_space.high, np.ones(4)])
        envs.single_observation_space = spaces.Box(low=low, high=high, shape=new_shape)

    agent = Agent(envs).to(device)

    best_goal_scored = 0

    if args.resume_from:
        resume_path = pathlib.Path(args.resume_from)
        agent = torch.load(
            resume_path / "agent.pth", map_location=device, weights_only=False
        )
        optimizer = optim.AdamW(
            agent.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        optimizer.load_state_dict(
            torch.load(resume_path / "optimizer.pth", map_location=device)
        )
        with open(resume_path / "global_step.txt") as f:
            global_step = int(f.read().strip())
        print(f"Resumed from {args.resume_from}, global_step = {global_step}")
    else:
        optimizer = optim.AdamW(
            agent.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    # ALGO Logic: Storage setup
    obs_shape = envs.single_observation_space.shape or ()
    obs = torch.zeros((args.num_steps, args.num_envs) + obs_shape).to(device)
    action_shape = envs.single_action_space.shape or ()
    actions = torch.zeros((args.num_steps, args.num_envs) + action_shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, infos = envs.reset(seed=args.seed)
    # Transform initial observations
    quats = infos["robot_quat"]
    next_obs = next_obs.copy()
    next_obs[:, 24:27] = quat_rotate_inverse(quats, next_obs[:, 24:27])  # ball pos
    next_obs[:, 27:30] = quat_rotate_inverse(quats, next_obs[:, 27:30])  # ball lin vel
    next_obs[:, 30:33] = quat_rotate_inverse(quats, next_obs[:, 30:33])  # ball ang vel
    next_obs[:, 33:36] = quat_rotate_inverse(quats, next_obs[:, 33:36])  # goal pos
    next_obs[:, 39:42] = quat_rotate_inverse(
        quats, next_obs[:, 39:42]
    )  # goalkeeper pos
    next_obs[:, 42:45] = quat_rotate_inverse(
        quats, next_obs[:, 42:45]
    )  # goalkeeper vel
    next_obs = np.concatenate([next_obs, quats], axis=1)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, terminations, truncations, infos = envs.step(
                action.cpu().numpy()
            )

            # Transform observations to robot frame
            quats = infos["robot_quat"]
            next_obs = next_obs.copy()
            next_obs[:, 24:27] = quat_rotate_inverse(
                quats, next_obs[:, 24:27]
            )  # ball pos
            next_obs[:, 27:30] = quat_rotate_inverse(
                quats, next_obs[:, 27:30]
            )  # ball lin vel
            next_obs[:, 30:33] = quat_rotate_inverse(
                quats, next_obs[:, 30:33]
            )  # ball ang vel
            next_obs[:, 33:36] = quat_rotate_inverse(
                quats, next_obs[:, 33:36]
            )  # target pos
            next_obs[:, 39:42] = quat_rotate_inverse(quats, next_obs[:, 39:42])
            next_obs[:, 42:45] = quat_rotate_inverse(quats, next_obs[:, 42:45])
            next_obs = np.concatenate([next_obs, quats], axis=1)

            next_done = terminations  # truncations are ignored for value bootstrapping
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = (
                torch.Tensor(next_obs).to(device),
                torch.Tensor(next_done).to(device),
            )

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = (
                    rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                )
                advantages[t] = lastgaelam = (
                    delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + values

        rewards_mean = rewards.mean().item()
        rewards_std = rewards.std().item()
        advantages_mean = advantages.mean().item()
        advantages_std = advantages.std().item()

        # flatten the batch
        b_obs = obs.reshape((-1,) + obs_shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [
                        ((ratio - 1.0).abs() > args.clip_coef).float().mean().item()
                    ]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - args.clip_coef, 1 + args.clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()  # type: ignore
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                # Check for NaN/inf in loss
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning(f"Loss is NaN/inf: {loss.item()}, skipping update")
                    optimizer.zero_grad()
                    continue

                optimizer.zero_grad()
                loss.backward()

                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        # Compute gradient stats once per iteration
        grad_norms = [
            torch.norm(p.grad).item() for p in agent.parameters() if p.grad is not None
        ]
        grad_max = max(grad_norms) if grad_norms else 0
        grad_mean = sum(grad_norms) / len(grad_norms) if grad_norms else 0

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        if args.track:
            wandb.log(
                {
                    "charts/learning_rate": optimizer.param_groups[0]["lr"],
                    "losses/value_loss": v_loss.item(),
                    "losses/policy_loss": pg_loss.item(),
                    "losses/entropy": entropy_loss.item(),
                    "losses/old_approx_kl": old_approx_kl.item(),
                    "losses/approx_kl": approx_kl.item(),
                    "losses/clipfrac": np.mean(clipfracs),
                    "losses/explained_variance": explained_var,
                    "gradients/max_norm": grad_max,
                    "gradients/mean_norm": grad_mean,
                    "charts/SPS": int(global_step / (time.time() - start_time)),
                    "charts/iteration": iteration,
                    "charts/global_step": global_step,
                    "rewards/mean": rewards_mean,
                    "rewards/std": rewards_std,
                    "advantages/mean": advantages_mean,
                    "advantages/std": advantages_std,
                },
                step=global_step,
            )

        print(f"Iteration {iteration} ")
        if iteration % args.eval_interval == 0:
            (
                rewards_mean,
                rewards_std,
                robot_fallen_total,
                offside_total,
                ball_blocked_total,
                goal_scored_total,
                ball_vel_twd_goal_mean,
                ball_vel_twd_goal_std,
                robot_distance_ball_mean,
                robot_distance_ball_std,
            ) = evaluate(
                agent, args.env_id, device, use_bc_controller=args.use_bc_controller
            )
            print(f"Mean reward: {rewards_mean}, Std: {rewards_std}")
            print(f"Total robot fallen events: {robot_fallen_total}")
            print(f"Total offside events: {offside_total}")
            print(f"Total ball blocked events: {ball_blocked_total}")
            print(f"Total goal scored events: {goal_scored_total}")
            print(f"Mean total ball_vel_twd_goal: {ball_vel_twd_goal_mean}")
            print(f"Mean total robot_distance_ball: {robot_distance_ball_mean}")
            if goal_scored_total > best_goal_scored:
                best_goal_scored = goal_scored_total
                torch.save(agent, pathlib.Path(checkpoint_dir) / "best_agent.pth")
                torch.save(
                    optimizer.state_dict(),
                    pathlib.Path(checkpoint_dir) / "best_optimizer.pth",
                )
                print(
                    f"New best goal scored: {best_goal_scored}, saved model to {checkpoint_dir}/best_agent.pth"
                )
            if args.track:
                wandb.log(
                    {
                        "eval/rewards_mean": rewards_mean,
                        "eval/rewards_std": rewards_std,
                        "eval/robot_fallen": robot_fallen_total,
                        "eval/offside": offside_total,
                        "eval/ball_blocked": ball_blocked_total,
                        "eval/goal_scored": goal_scored_total,
                        "eval/ball_vel_twd_goal_mean": ball_vel_twd_goal_mean,
                        "eval/ball_vel_twd_goal_std": ball_vel_twd_goal_std,
                        "eval/robot_distance_ball_mean": robot_distance_ball_mean,
                        "eval/robot_distance_ball_std": robot_distance_ball_std,
                    },
                    step=global_step,
                )

        if iteration % args.checkpoint_interval == 0:
            ckpt_path = pathlib.Path(checkpoint_dir) / str(iteration)
            ckpt_path.mkdir(exist_ok=True)
            torch.save(agent, ckpt_path / "agent.pth")
            torch.save(optimizer.state_dict(), ckpt_path / "optimizer.pth")
            import json

            with open(ckpt_path / "args.json", "w") as f:
                json.dump(vars(args), f)
            with open(ckpt_path / "global_step.txt", "w") as f:
                f.write(str(global_step))

    envs.close()
