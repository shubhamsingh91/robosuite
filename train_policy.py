"""
Simple Policy Training for Robosuite Lift Task
===============================================
Uses PPO from stable-baselines3 with state observations (no images).
Designed to run on CPU.
"""

import gymnasium as gym
import numpy as np
import time
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
import torch
import os
import matplotlib.pyplot as plt


class PolicyStatsCallback(BaseCallback):
    """Callback to print policy mean and std during training."""

    def __init__(self, print_freq=512, verbose=0):
        super().__init__(verbose)
        self.print_freq = print_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.print_freq == 0:
            # Get the policy network
            policy = self.model.policy

            # Get current observation
            obs = self.locals.get("obs_tensor")
            if obs is None:
                obs = torch.tensor(self.training_env.buf_obs[None]).float()

            # Get action distribution from policy
            with torch.no_grad():
                obs_tensor = torch.as_tensor(self.locals["new_obs"]).float().to(self.model.device)
                distribution = policy.get_distribution(obs_tensor)
                mean = distribution.distribution.mean.cpu().numpy()
                std = distribution.distribution.stddev.cpu().numpy()

            # Print stats
            print(f"\n--- Step {self.num_timesteps} ---")
            print(f"Action means: [{', '.join(f'{m:.3f}' for m in mean[0])}]")
            print(f"Action stds:  [{', '.join(f'{s:.3f}' for s in std[0])}]")

        return True


class LivePlotCallback(BaseCallback):
    """Callback to show live plot of training losses and action std."""

    def __init__(self, plot_freq=512, verbose=0):
        super().__init__(verbose)
        self.plot_freq = plot_freq
        self.policy_losses = []
        self.value_losses = []
        self.rewards = []
        self.episode_lengths = []  # Track episode lengths
        self.action_stds = []  # Average std across all action dimensions
        self.action_stds_per_dim = [[] for _ in range(7)]  # Std per action dimension
        self.timesteps = []
        self.fig = None
        self.axes = None
        # Track episode rewards and lengths directly
        self.episode_rewards = []
        self.episode_lens = []
        self.current_episode_reward = 0
        self.current_episode_len = 0

    def _on_training_start(self):
        plt.ion()
        self.fig, self.axes = plt.subplots(2, 3, figsize=(15, 8))
        self.axes = self.axes.flatten()
        self.fig.suptitle('PPO Training Progress')
        self.axes[0].set_ylabel('Policy Loss')
        self.axes[1].set_ylabel('Value Loss')
        self.axes[2].set_ylabel('Episode Reward')
        self.axes[3].set_ylabel('Episode Length')
        self.axes[4].set_ylabel('Action Std (σ)')
        self.axes[5].axis('off')  # Empty panel
        for ax in self.axes[:5]:
            ax.set_xlabel('Timesteps')
            ax.grid(True)
        plt.tight_layout()
        plt.show(block=False)

    def _on_step(self) -> bool:
        # Track rewards and lengths from each step
        rewards = self.locals.get('rewards', [])
        dones = self.locals.get('dones', [])

        if len(rewards) > 0:
            self.current_episode_reward += rewards[0]
            self.current_episode_len += 1
            if dones[0]:
                self.episode_rewards.append(self.current_episode_reward)
                self.episode_lens.append(self.current_episode_len)
                self.current_episode_reward = 0
                self.current_episode_len = 0
        return True

    def _on_rollout_end(self):
        # Get losses from logger
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            logs = self.logger.name_to_value
            policy_loss = logs.get('train/policy_gradient_loss', None)
            value_loss = logs.get('train/value_loss', None)
            # Try multiple possible reward keys
            reward = logs.get('rollout/ep_rew_mean', None)
            if reward is None:
                reward = logs.get('ep_rew_mean', None)
            if reward is None:
                reward = logs.get('train/ep_rew_mean', None)

            # Get current action std from policy
            policy = self.model.policy
            std = None
            with torch.no_grad():
                # Best approach: get distribution with a dummy observation
                try:
                    dummy_obs = torch.zeros(1, self.model.observation_space.shape[0]).to(self.model.device)
                    dist = policy.get_distribution(dummy_obs)
                    std = dist.distribution.stddev.cpu().numpy()[0]
                except Exception:
                    pass

                # Fallback: try to access log_std directly
                if std is None:
                    try:
                        if hasattr(policy, 'log_std') and policy.log_std is not None:
                            log_std = policy.log_std.cpu().numpy()
                            std = np.exp(log_std)
                    except Exception:
                        pass

                # Final fallback
                if std is None:
                    std = np.ones(7) * 0.5

            if policy_loss is not None:
                self.timesteps.append(self.num_timesteps)
                self.policy_losses.append(policy_loss)
                self.value_losses.append(value_loss)

                # Use tracked episode rewards if logger reward not available
                if reward is not None:
                    self.rewards.append(reward)
                elif len(self.episode_rewards) > 0:
                    # Use average of recent episodes
                    recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) > 10 else self.episode_rewards
                    self.rewards.append(np.mean(recent_rewards))
                elif len(self.rewards) > 0:
                    self.rewards.append(self.rewards[-1])  # Keep last value
                else:
                    self.rewards.append(0)

                # Track episode length
                if len(self.episode_lens) > 0:
                    recent_lens = self.episode_lens[-10:] if len(self.episode_lens) > 10 else self.episode_lens
                    self.episode_lengths.append(np.mean(recent_lens))
                elif len(self.episode_lengths) > 0:
                    self.episode_lengths.append(self.episode_lengths[-1])
                else:
                    self.episode_lengths.append(200)  # Default to max horizon

                # Store average std and per-dimension std
                self.action_stds.append(np.mean(std))
                for i in range(min(7, len(std))):
                    self.action_stds_per_dim[i].append(std[i])

                # Update plots
                for i in range(5):
                    self.axes[i].clear()

                self.axes[0].plot(self.timesteps, self.policy_losses, 'b-')
                self.axes[1].plot(self.timesteps, self.value_losses, 'r-')
                self.axes[2].plot(self.timesteps, self.rewards, 'g-')
                self.axes[3].plot(self.timesteps, self.episode_lengths, 'm-')

                # Plot all 7 action stds with different colors
                colors = ['b', 'orange', 'g', 'r', 'purple', 'brown', 'pink']
                labels = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Grip']
                for i in range(7):
                    if len(self.action_stds_per_dim[i]) > 0:
                        self.axes[4].plot(self.timesteps, self.action_stds_per_dim[i],
                                         color=colors[i], label=labels[i], alpha=0.7)
                self.axes[4].legend(loc='upper right', fontsize=8)

                self.axes[0].set_ylabel('Policy Loss')
                self.axes[1].set_ylabel('Value Loss')
                self.axes[2].set_ylabel('Episode Reward')
                self.axes[3].set_ylabel('Episode Length')
                self.axes[4].set_ylabel('Action Std (σ)')

                for ax in self.axes[:5]:
                    ax.set_xlabel('Timesteps')
                    ax.grid(True)

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    def _on_training_end(self):
        plt.ioff()
        plt.savefig('training_losses.png')
        print("\nLoss plot saved to training_losses.png")


class RobosuiteGymEnv(gym.Env):
    """
    Minimal Gymnasium wrapper for robosuite.
    Uses only low-dimensional state (no images) for fast CPU training.
    """

    def __init__(self, render_mode=None):
        super().__init__()

        # Create robosuite environment
        self.env = suite.make(
            env_name="Lift",
            robots="Panda",
            has_renderer=False,  # Show GUI during training
            has_offscreen_renderer=False, # used for camera obsm if true, then we use camera images for training
            use_camera_obs=False,  # No images = faster
            horizon=200,  # Longer episodes for better learning
            control_freq=20,
            reward_shaping=True,  # Dense rewards help learning
        )

        # Get actual observation dimensions from environment
        obs_dict = self.env.reset()
        sample_obs = self._get_obs_internal(obs_dict)
        obs_dim = sample_obs.shape[0]
        print(f"Observation dimension: {obs_dim}")

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: 7D (6 OSC + 1 gripper)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(7,), dtype=np.float32
        )

        self.render_mode = render_mode

    def _get_obs_internal(self, obs_dict):
        """Extract flat observation from robosuite obs dict."""
        return np.concatenate([
            obs_dict["robot0_proprio-state"],
            obs_dict["object-state"],
        ]).astype(np.float32)

    def _get_obs(self, obs_dict):
        """Extract flat observation from robosuite obs dict."""
        return self._get_obs_internal(obs_dict)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        obs_dict = self.env.reset()
        return self._get_obs(obs_dict), {}

    def step(self, action):
        obs_dict, reward, done, info = self.env.step(action)
        obs = self._get_obs(obs_dict)

        # Manually check success since robosuite doesn't populate info dict
        success = self.env._check_success()
        info["success"] = success

        # Add lifting reward: bonus for cube height above table
        cube_height = obs_dict["cube_pos"][2]
        table_height = 0.8
        lift_height = cube_height - table_height
        if lift_height > 0.01:  # Cube lifted at least 1cm
            reward += lift_height * 10.0  # Reward proportional to height

        # If success, terminate early with done=True
        if success:
            done = True
            reward += 50.0  # Large bonus reward for success

        # Handle truncation (horizon reached without success)
        truncated = False
        if done and not success:
            truncated = True
            done = False

        return obs, reward, done, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


def make_env():
    """Factory function for creating the environment."""
    return RobosuiteGymEnv()


def train(device="cuda", total_timesteps=50_000):
    """Train a PPO policy on the Lift task."""

    print("Creating environment...")
    env = make_env()

    # Create output directory
    os.makedirs("trained_models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    # Callbacks for saving
    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./trained_models/",
        name_prefix="ppo_lift"
    )

    print(f"Creating PPO model ({device.upper()})...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,  # Larger buffer for more stable updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.001,  # Small entropy bonus (lower = faster σ convergence)
        clip_range=0.2,
        policy_kwargs=dict(net_arch=[256, 256]),  # Larger network
        device=device,
    )

    print("Starting training...")
    print(f"Training on {device.upper()}. Press Ctrl+C to stop early.")
    print("-" * 50)

    # Create callbacks
    stats_callback = PolicyStatsCallback(print_freq=512)
    live_plot_callback = LivePlotCallback()

    try:
        # Train for a modest number of steps (increase for better results)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, stats_callback, live_plot_callback],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    model.save("trained_models/ppo_lift_final")
    print("\nModel saved to trained_models/ppo_lift_final.zip")

    env.close()
    return model


def test_trained_policy(model_path="trained_models/ppo_lift_final", plot_trajectories=True):
    """Test the trained policy with visualization and trajectory plotting."""

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)

    # Create env WITH rendering for visualization
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        has_renderer=True,  # Show GUI
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=200,
        control_freq=20,
        reward_shaping=True,
    )

    print("Running trained policy (close window to stop)...")

    # Storage for trajectory data (last episode)
    trajectory_data = {
        'time': [],
        'joint_pos': [],
        'joint_vel': [],
        'joint_torque': [],
        'ee_pos': [],
        'ee_ori': [],
        'actions': [],
        'gripper_pos': [],
        'cube_pos': [],
        'rewards': [],
    }

    dt = 1.0 / 20  # control_freq = 20 Hz

    for episode in range(5):
        obs_dict = env.reset()
        obs = np.concatenate([
            obs_dict["robot0_proprio-state"],
            obs_dict["object-state"],
        ]).astype(np.float32)

        # Clear trajectory data for new episode
        for key in trajectory_data:
            trajectory_data[key] = []

        total_reward = 0
        done = False
        step = 0

        while not done and step < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs_dict, reward, done, info = env.step(action)

            # Record trajectory data
            trajectory_data['time'].append(step * dt)
            trajectory_data['actions'].append(action.copy())
            trajectory_data['rewards'].append(reward)

            # Get robot state from environment
            robot = env.robots[0]
            trajectory_data['joint_pos'].append(robot.sim.data.qpos[robot._ref_joint_pos_indexes].copy())
            trajectory_data['joint_vel'].append(robot.sim.data.qvel[robot._ref_joint_vel_indexes].copy())
            trajectory_data['joint_torque'].append(robot.sim.data.ctrl[robot._ref_arm_joint_actuator_indexes].copy())
            # Get EE position using the site id (eef_site_id is a dict with 'right' key for Panda)
            eef_site_idx = robot.eef_site_id['right']
            trajectory_data['ee_pos'].append(robot.sim.data.site_xpos[eef_site_idx].copy())
            trajectory_data['gripper_pos'].append(robot._hand_pos.copy())

            # Get cube position
            cube_pos = obs_dict.get("cube_pos", obs_dict.get("object-state", np.zeros(3))[:3])
            trajectory_data['cube_pos'].append(cube_pos.copy() if hasattr(cube_pos, 'copy') else cube_pos)

            obs = np.concatenate([
                obs_dict["robot0_proprio-state"],
                obs_dict["object-state"],
            ]).astype(np.float32)

            total_reward += reward
            step += 1
            env.render()
            time.sleep(0.05)  # Slow down visualization (50ms per step)

        success = "SUCCESS" if info.get("success", False) else "FAILED"
        print(f"Episode {episode + 1}: {success}, reward={total_reward:.2f}, steps={step}")

    env.close()

    # Plot trajectories from last episode
    if plot_trajectories and len(trajectory_data['time']) > 0:
        plot_trajectory_data(trajectory_data)


def plot_trajectory_data(data):
    """Plot joint positions, torques, actions, and end-effector trajectory."""
    time = np.array(data['time'])
    joint_pos = np.array(data['joint_pos'])
    joint_torque = np.array(data['joint_torque'])
    actions = np.array(data['actions'])
    ee_pos = np.array(data['ee_pos'])
    rewards = np.array(data['rewards'])

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Policy Execution Trajectory (Last Episode)', fontsize=14)

    # Joint Positions
    ax = axes[0, 0]
    for i in range(min(7, joint_pos.shape[1])):
        ax.plot(time, joint_pos[:, i], label=f'Joint {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Joint Position (rad)')
    ax.set_title('Joint Positions')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

    # Joint Torques
    ax = axes[0, 1]
    for i in range(min(7, joint_torque.shape[1])):
        ax.plot(time, joint_torque[:, i], label=f'Joint {i+1}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Torque (Nm)')
    ax.set_title('Joint Torques (Control Commands)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)

    # Policy Actions
    ax = axes[1, 0]
    action_names = ['X', 'Y', 'Z', 'Roll', 'Pitch', 'Yaw', 'Grip']
    for i in range(min(7, actions.shape[1])):
        ax.plot(time, actions[:, i], label=action_names[i])
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Action Value')
    ax.set_title('Policy Actions (OSC Commands)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True)
    ax.set_ylim(-1.1, 1.1)

    # End-Effector Position
    ax = axes[1, 1]
    ax.plot(time, ee_pos[:, 0], 'r-', label='X')
    ax.plot(time, ee_pos[:, 1], 'g-', label='Y')
    ax.plot(time, ee_pos[:, 2], 'b-', label='Z')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (m)')
    ax.set_title('End-Effector Position')
    ax.legend(loc='upper right')
    ax.grid(True)

    # Rewards over time
    ax = axes[2, 0]
    ax.plot(time, rewards, 'g-')
    ax.plot(time, np.cumsum(rewards), 'b--', label='Cumulative')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Over Time')
    ax.legend()
    ax.grid(True)

    # 3D End-Effector Trajectory
    ax = axes[2, 1]
    ax.remove()
    ax = fig.add_subplot(3, 2, 6, projection='3d')
    ax.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], 'b-', linewidth=2)
    ax.scatter(ee_pos[0, 0], ee_pos[0, 1], ee_pos[0, 2], c='g', s=100, marker='o', label='Start')
    ax.scatter(ee_pos[-1, 0], ee_pos[-1, 1], ee_pos[-1, 2], c='r', s=100, marker='x', label='End')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('End-Effector 3D Trajectory')
    ax.legend()

    plt.tight_layout()
    plt.savefig('trajectory_plot.png')
    print("\nTrajectory plot saved to trajectory_plot.png")
    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train or test PPO policy on Robosuite Lift task")
    parser.add_argument("--test", action="store_true", help="Test a trained policy")
    parser.add_argument("--model", type=str, default="trained_models/ppo_lift_final",
                        help="Path to model for testing (default: trained_models/ppo_lift_final)")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for training")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training (default)")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total timesteps for training (default: 50000)")
    args = parser.parse_args()

    if args.test:
        test_trained_policy(model_path=args.model)
    else:
        device = "cpu" if args.cpu else "cuda"
        model = train(device=device, total_timesteps=args.timesteps)
        print("\nTo test the trained policy, run:")
        print("  python train_policy.py --test")
