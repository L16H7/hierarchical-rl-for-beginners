import os
import sys
import argparse
import gymnasium as gym
import torch
import glfw
import time
import imageio
import numpy as np
from ppo.train import (
    CommandActionWrapper,
    Agent,
    quat_rotate_inverse,
)

# Make repo root importable without absolute paths
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)


def load_ppo_agent(checkpoint_path: str, device: str = "cpu") -> Agent:
    """Load a PPO agent from checkpoint"""
    model_path = os.path.join(checkpoint_path, "agent.pth")
    agent = torch.load(model_path, map_location=device, weights_only=False)
    agent.eval()
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env",
        type=str,
        default="LowerT1GoaliePenaltyKick-v0",
        help="Gym env ID (e.g., LowerT1GoaliePenaltyKick-v0)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (containing args.json and agent.pth)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="PyTorch device (e.g., 'cpu', 'mps', 'cuda')",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of rollout episodes",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save videos instead of watching live",
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="videos",
        help="Directory to save videos (default: videos)",
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback speed multiplier for live viewing (default: 1.0)",
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Skip every N frames for faster playback/recording (default: 1)",
    )
    args = parser.parse_args()

    # Set checkpoint path
    checkpoint_path = args.checkpoint
    print(f"[INFO] Loading checkpoint: {checkpoint_path}")

    # Load the PPO agent
    try:
        agent = load_ppo_agent(checkpoint_path, args.device)
        print("[INFO] Successfully loaded PPO agent")
    except Exception as e:
        print(f"[ERROR] Failed to load agent: {e}")
        sys.exit(1)

    # Create video directory if saving videos
    if args.save_video:
        os.makedirs(args.video_dir, exist_ok=True)
        print(f"[INFO] Videos will be saved to: {args.video_dir}")

    # Build env with appropriate render mode
    render_mode = "rgb_array" if args.save_video else "human"

    base_env = gym.make(args.env, render_mode=render_mode)
    env = CommandActionWrapper(base_env)

    # ---- Rollout ----
    ep_goal_scored = 0
    for ep in range(args.episodes):
        obs, info = env.reset(seed=(424 + ep))

        # Transform observation to robot frame as in training
        if "robot_quat" in info:
            quat = info["robot_quat"]
            obs = obs.copy()
            obs[24:27] = quat_rotate_inverse(quat, obs[24:27])  # ball pos
            obs[27:30] = quat_rotate_inverse(quat, obs[27:30])  # ball lin vel
            obs[30:33] = quat_rotate_inverse(quat, obs[30:33])  # ball ang vel
            obs[33:36] = quat_rotate_inverse(quat, obs[33:36])  # target pos
            obs[39:42] = quat_rotate_inverse(quat, obs[39:42])  # goalkeeper pos
            obs[42:45] = quat_rotate_inverse(quat, obs[42:45])  # goalkeeper vel
            obs = np.concatenate([obs, quat])
        else:
            raise ValueError("robot_quat not found in info during reset.")

        terminated = truncated = False
        ep_return = 0.0
        step_count = 0
        frames = []
        ball_vel_history = []
        robot_dist_history = []

        mode_str = "Recording" if args.save_video else "Playing"
        print(f"[Episode {ep + 1}] {mode_str}. Press ESC to stop (live mode only).")

        while not (terminated or truncated):
            # Check for window events if in live mode
            if not args.save_video:
                try:
                    # Try to get viewer for ESC key detection
                    if hasattr(env, "unwrapped") and hasattr(
                        env.unwrapped, "mujoco_renderer"
                    ):
                        viewer = getattr(env.unwrapped.mujoco_renderer, "viewer", None)  # type: ignore
                        window = (
                            getattr(viewer, "window", None)
                            if viewer is not None
                            else None
                        )

                        if (
                            window is not None
                            and glfw.get_current_context() is not None
                        ):
                            # Stop if user hit ESC inside the MuJoCo window
                            if glfw.get_key(window, glfw.KEY_ESCAPE) == glfw.PRESS:
                                print("\n[INFO] ESC pressed — stopping and closing.")
                                env.close()
                                sys.exit(0)

                            # Stop if user clicked the window close button (red X)
                            if glfw.window_should_close(window):
                                print("\n[INFO] Window closed — stopping and exiting.")
                                env.close()
                                sys.exit(0)
                except Exception:
                    # Ignore viewer errors
                    pass

            # Get action from PPO agent
            with torch.no_grad():
                obs_tensor = torch.Tensor(obs).unsqueeze(0).to(args.device)
                action = agent.get_action(obs_tensor)
                # import pdb; pdb.set_trace()
                action = action.cpu().numpy()[0]

            print(f"[STEP {step_count + 1}] action: {action}")
            skipper_rewards = 0.0
            skipper_info = {
                "robot_fallen": 0,
                "offside": 0,
                "ball_blocked": 0,
                "goal_scored": 0,
                "ball_vel_twd_goal": 0,
                "robot_distance_ball": 0,
            }
            for _ in range(50):
                obs, _, terminated, truncated, info = env.step(action)
                step_count += 1

                # Transform observation after step
                if "robot_quat" in info:
                    quat = info["robot_quat"]
                    obs = obs.copy()
                    obs[24:27] = quat_rotate_inverse(quat, obs[24:27])  # ball pos
                    obs[27:30] = quat_rotate_inverse(quat, obs[27:30])  # ball lin vel
                    obs[30:33] = quat_rotate_inverse(quat, obs[30:33])  # ball ang vel
                    obs[33:36] = quat_rotate_inverse(quat, obs[33:36])  # target pos
                    obs[39:42] = quat_rotate_inverse(quat, obs[39:42])  # goalkeeper pos
                    obs[42:45] = quat_rotate_inverse(quat, obs[42:45])  # goalkeeper vel
                    obs = np.concatenate([obs, quat])
                else:
                    raise ValueError("robot_quat not found in info during step.")

                from ppo.train import get_reward_and_info

                reward, step_info = get_reward_and_info(info.get("reward_terms", {}))
                skipper_rewards += float(reward)
                skipper_info = {
                    key: skipper_info[key] + step_info[key] for key in skipper_info
                }

                # Handle frame recording or speed control
                if args.save_video:
                    # Record frame for video
                    if step_count % args.skip_frames == 0:
                        frame = env.render()
                        if frame is not None:
                            frames.append(frame)
                else:
                    # Control playback speed for live viewing
                    if args.speed < 10.0:  # Only add delay for reasonable speeds
                        time.sleep(
                            max(0, (1.0 / args.speed - 1.0) * 0.016)
                        )  # ~60 FPS base

                ep_goal_scored += skipper_info["goal_scored"]
                if terminated or truncated:
                    # import pdb; pdb.set_trace()
                    break
            print(
                f"[STEP {step_count}] skipper_rewards: {skipper_rewards}, goal_scored in block: {skipper_info['goal_scored']}"
            )

        print(
            f"[Episode {ep + 1}] return = {ep_return:.3f}, total goal_scored: {ep_goal_scored}"
        )

        # Save video if recording
        if args.save_video and frames:
            video_filename = f"episode_{ep + 1:03d}.mp4"
            video_path = os.path.join(args.video_dir, video_filename)

            try:
                # Calculate fps based on skip_frames (assuming base 60 fps)
                fps = max(1, 60 // args.skip_frames)
                imageio.mimsave(video_path, frames, fps=fps)
                print(
                    f"[INFO] Saved video: {video_path} ({len(frames)} frames, {fps} fps)"
                )
            except Exception as e:
                print(f"[ERROR] Failed to save video {video_path}: {e}")

    env.close()
    print("[INFO] Environment closed. Exiting.")


if __name__ == "__main__":
    main()
