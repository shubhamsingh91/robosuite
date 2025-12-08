# Robosuite Lift Task

A robotic manipulation project using [robosuite](https://robosuite.ai/) for simulation and PPO for policy learning.

## Setup

### 1. Create conda environment

```bash
conda create -n robosuite python=3.10 -y
conda activate robosuite
```

### 2. Install dependencies

```bash
pip install robosuite numpy opencv-python stable-baselines3 gymnasium h5py tqdm rich matplotlib
```

## Project Structure

```
.
├── pick_place_example.py   # Demo script with camera rendering
├── train_policy.py         # PPO training script
├── inspect_policy.py       # Interactive policy inspector
├── trained_models/         # Saved policy checkpoints
├── camera_images/          # Captured camera frames
├── training_losses.png     # Loss plot from training
├── trajectory_plot.png     # Trajectory plot from testing
└── README.md
```

## Usage

### Train a policy

```bash
python train_policy.py --timesteps 500000
```

**Command line options:**
| Option | Description |
|--------|-------------|
| `--gpu` | Use GPU for training (default) |
| `--cpu` | Use CPU for training |
| `--timesteps N` | Total training timesteps (default: 50,000) |
| `--test` | Test a trained policy |
| `--model PATH` | Path to model for testing |

**Examples:**
```bash
# Train with 500k steps
python train_policy.py --timesteps 500000

# Train on CPU
python train_policy.py --cpu --timesteps 100000

# Train for 1M+ steps for better success rate
python train_policy.py --timesteps 1000000
```

**Training parameters (in code):**
- `n_steps`: Steps before each policy update (default: 2048)
- `n_epochs`: Passes through collected data per update (default: 10)
- `batch_size`: Batch size for updates (default: 64)
- `learning_rate`: Learning rate (default: 3e-4)
- `ent_coef`: Entropy coefficient (default: 0.001)
- `horizon`: Max steps per episode (default: 200)

**Live monitoring:**
- Progress bar shows training progress
- Live plot window displays:
  - Policy Loss
  - Value Loss
  - Episode Reward
  - Episode Length
  - Action Std (per dimension)
- Loss plot saved to `training_losses.png` when training ends

### Test trained policy

```bash
# Test the final model
python train_policy.py --test

# Test a specific checkpoint
python train_policy.py --test --model trained_models/ppo_lift_420000_steps
```

Runs 5 episodes with visualization and generates trajectory plots showing:
- Joint positions over time
- Joint torques (control commands)
- Policy actions (OSC commands)
- End-effector position
- Rewards over time
- 3D end-effector trajectory

### Inspect policy

```bash
python inspect_policy.py --model trained_models/ppo_lift_500000_steps
```

Interactive tool to query the policy with observations and visualize action distributions.

## Key Concepts

### PPO Training

The policy is a neural network that maps observations to actions:
- **Policy**: `π(a|o) = N(μ(o), σ²)` - Gaussian distribution over actions
- **Value function**: `V(o)` - Predicts expected future reward
- **Policy Loss**: Clipped surrogate objective (should be negative, closer to 0 is better)
- **Value Loss**: MSE between predicted and actual returns (should decrease)

### Environment Settings

| Parameter | Description |
|-----------|-------------|
| `has_renderer` | Show GUI window |
| `has_offscreen_renderer` | Enable off-screen rendering for camera images |
| `use_camera_obs` | Include camera images in observations |
| `horizon` | Max steps per episode (200) |
| `control_freq` | Control frequency in Hz (20) |
| `reward_shaping` | Use dense rewards (easier to learn) |

### Observation Space (~53D)

**State-based (for training):**
- `robot0_proprio-state` (~43D): Joint positions, velocities, end-effector pose
- `object-state` (~10D): Cube position, orientation, gripper-to-cube distance

### Action Space (7D)

Continuous actions for OSC (Operational Space Control):
| Dimension | Description | Range |
|-----------|-------------|-------|
| 0-2 | End-effector linear velocity (x, y, z) | [-1, 1] |
| 3-5 | End-effector angular velocity (roll, pitch, yaw) | [-1, 1] |
| 6 | Gripper command (-1=open, +1=close) | [-1, 1] |

### Reward Structure (Lift Task)

Dense reward shaping with components:
1. **Reaching**: Reward for moving gripper closer to cube
2. **Grasping**: Reward for closing gripper on cube
3. **Lifting**: Reward for lifting cube above threshold
4. **Success bonus**: Large reward when cube reaches target height

## Training Tips

1. **Train longer**: 500k steps gets ~70 reward but may not achieve success. Try 1M+ steps.
2. **Monitor action std (σ)**: Should decrease from ~1.0 to ~0.2-0.3 for confident actions
3. **Watch episode reward**: Should increase; plateau around 70-80 means reaching but not lifting
4. **Check gripper action**: If gripper σ stays high, the policy isn't learning when to grasp

## Troubleshooting

**Policy not achieving success:**
- Train for more timesteps (1M+)
- Action std (σ) may still be too high (>0.5)
- Policy may be stuck in local optimum (reaching but not grasping)

**Rewards plateau:**
- Try different `ent_coef` values (0.0001 for faster convergence, 0.01 for more exploration)
- Increase `n_steps` for more stable updates
- Use larger network (`net_arch=[256, 256]` or `[512, 512]`)

## Sim-to-Real

The current training uses ground-truth state (object positions from simulator). For real robot deployment:

1. Set `use_camera_obs=True`
2. Use CNN-based policy instead of MLP
3. Add domain randomization (textures, lighting, camera angles)
4. Train much longer (requires GPU)
