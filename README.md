### Teleoperator

```python
python ./booster_control/teleoperate.py --env LowerT1GoaliePenaltyKick-v0 --pos_sensitivity 1.5 --rot_sensitivity 1.5
```

### PPO Training

```python
python -m ppo.train --seed 2026 --env_id LowerT1GoaliePenaltyKick-v0 --wandb_project_name ppo_goalie_penalty_kick_skipper --wandb_entity l16h7 --num_envs 100 --total_timesteps 100_000_000 --num_minibatches 20 --update_epochs 1 --learning_rate 3e-4 --eval_interval 8 --checkpoint-interval 40 --num_steps 50 --track
```
