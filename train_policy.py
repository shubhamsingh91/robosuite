"""
Simple Policy Training for Robosuite Lift Task
===============================================
Uses PPO from stable-baselines3 with state observations (no images).
Designed to run on CPU.
"""

import gymnasium as gym
import numpy as np
import robosuite as suite
from robosuite.wrappers import GymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
import os


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
            has_renderer=True,  # Show GUI during training
            has_offscreen_renderer=False, # used for camera obsm if true, then we use camera images for training
            use_camera_obs=False,  # No images = faster
            horizon=100,  # Shorter episodes for faster training
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

        # Check for success
        truncated = False
        if done and not info.get("success", False):
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


def train():
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

    print("Creating PPO model (CPU)...")
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=512,  # Smaller buffer for faster updates
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        device="cpu",  # Force CPU
    )

    print("Starting training...")
    print("This will take a while on CPU. Press Ctrl+C to stop early.")
    print("-" * 50)

    try:
        # Train for a modest number of steps (increase for better results)
        model.learn(
            total_timesteps=50_000,  # Adjust based on patience
            callback=checkpoint_callback,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")

    # Save final model
    model.save("trained_models/ppo_lift_final")
    print("\nModel saved to trained_models/ppo_lift_final.zip")

    env.close()
    return model


def test_trained_policy(model_path="trained_models/ppo_lift_final"):
    """Test the trained policy with visualization."""

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

    for episode in range(5):
        obs_dict = env.reset()
        obs = np.concatenate([
            obs_dict["robot0_proprio-state"],
            obs_dict["object-state"],
        ]).astype(np.float32)

        total_reward = 0
        done = False
        step = 0

        while not done and step < 200:
            action, _ = model.predict(obs, deterministic=True)
            obs_dict, reward, done, info = env.step(action)
            obs = np.concatenate([
                obs_dict["robot0_proprio-state"],
                obs_dict["object-state"],
            ]).astype(np.float32)

            total_reward += reward
            step += 1
            env.render()

        success = "SUCCESS" if info.get("success", False) else "FAILED"
        print(f"Episode {episode + 1}: {success}, reward={total_reward:.2f}, steps={step}")

    env.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_trained_policy()
    else:
        model = train()
        print("\nTo test the trained policy, run:")
        print("  python train_policy.py test")
