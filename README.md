# Robosuite Pick and Place

A simple robotic manipulation project using [robosuite](https://robosuite.ai/) for simulation and PPO for policy learning.

## Setup

### 1. Create conda environment

```bash
conda create -n robosuite python=3.10 -y
conda activate robosuite
```

### 2. Install dependencies

```bash
pip install robosuite numpy opencv-python stable-baselines3 gymnasium h5py tqdm rich
```

## Project Structure

```
.
├── pick_place_example.py   # Demo script with camera rendering
├── train_policy.py         # PPO training script
├── trained_models/         # Saved policy checkpoints
├── camera_images/          # Captured camera frames
├── training_losses.png     # Loss plot from training
└── README.md
```

## Usage

### Run demo with random actions

```bash
python pick_place_example.py
```

Opens a GUI showing the Panda robot with random actions. Captures camera images from:
- `agentview` - Third-person camera
- `robot0_eye_in_hand` - Wrist-mounted camera

### Train a policy

```bash
python train_policy.py --gpu --timesteps 500000
```

**Command line options:**
| Option | Description |
|--------|-------------|
| `--gpu` | Use GPU for training (default) |
| `--cpu` | Use CPU for training |
| `--timesteps N` | Total training timesteps (default: 50,000) |
| `--test` | Test a trained policy |

**Examples:**
```bash
# Train on GPU with 500k steps
python train_policy.py --gpu --timesteps 500000

# Train on CPU with default steps
python train_policy.py --cpu

# Quick test run
python train_policy.py --timesteps 10000
```

**Training parameters (in code):**
- `n_steps`: Steps before each policy update (default: 512)
- `n_epochs`: Passes through collected data per update (default: 10)
- `batch_size`: Batch size for updates (default: 64)
- `learning_rate`: Learning rate (default: 3e-4)
- `horizon`: Max steps per episode (default: 100)

**Live monitoring:**
- Progress bar shows training progress
- Live plot window displays policy loss, value loss, and entropy loss
- Action means/stds printed every 512 steps
- Loss plot saved to `training_losses.png` when training ends

### Test trained policy

```bash
python train_policy.py --test
```

Runs 5 episodes with the trained policy and displays results.

## Key Concepts

### Environment Settings

| Parameter | Description |
|-----------|-------------|
| `has_renderer` | Show GUI window |
| `has_offscreen_renderer` | Enable off-screen rendering for camera images |
| `use_camera_obs` | Include camera images in observations |
| `horizon` | Max steps per episode |
| `control_freq` | Control frequency in Hz |
| `reward_shaping` | Use dense rewards (easier to learn) |

### Observation Space

**State-based (for training):**
- `robot0_proprio-state` (~43D): Joint positions, velocities, end-effector pose
- `object-state` (~10D): Cube position, orientation, gripper-to-cube distance

**Vision-based (for sim-to-real):**
- `agentview_image`: (256, 256, 3) RGB
- `robot0_eye_in_hand_image`: (256, 256, 3) RGB

### Action Space

7D continuous actions (OSC controller):
- Dimensions 0-2: End-effector linear velocity (x, y, z)
- Dimensions 3-5: End-effector angular velocity (roll, pitch, yaw)
- Dimension 6: Gripper command (-1=open, +1=close)

## Training Tips

1. **Start small**: Use 50k-100k timesteps to verify things work
2. **Train longer for success**: 200k-500k steps typically needed for reliable Lift success
3. **Monitor rewards**: Should increase over training
4. **Watch std decrease**: Policy becomes more confident as training progresses

## Sim-to-Real

The current training uses ground-truth state (object positions from simulator). For real robot deployment:

1. Set `use_camera_obs=True`
2. Use CNN-based policy instead of MLP
3. Add domain randomization (textures, lighting, camera angles)
4. Train much longer (requires GPU)
