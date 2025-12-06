"""
Interactive Policy Inspector
============================
Load a trained PPO policy and query it with observations.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from stable_baselines3 import PPO


def load_policy(model_path="trained_models/ppo_lift_final"):
    """Load a trained PPO model."""
    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print(f"Model loaded successfully!")
    print(f"  Observation space: {model.observation_space.shape}")
    print(f"  Action space: {model.action_space.shape}")
    return model


def get_action_distribution(model, obs):
    """Get the full action distribution (mean and std) for an observation."""
    obs_tensor = torch.as_tensor(obs).float().unsqueeze(0).to(model.device)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        mean = distribution.distribution.mean.cpu().numpy()[0]
        std = distribution.distribution.stddev.cpu().numpy()[0]

    return mean, std


def get_action(model, obs, deterministic=True):
    """Get action for an observation."""
    action, _ = model.predict(obs, deterministic=deterministic)
    return action


def plot_action_distributions(mean, std, show=True, save_path=None):
    """Plot the 7 Gaussian distributions for each action dimension."""
    action_names = [
        "EE vel X", "EE vel Y", "EE vel Z",
        "EE ang vel (roll)", "EE ang vel (pitch)", "EE ang vel (yaw)",
        "Gripper"
    ]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    axes = axes.flatten()

    # X range for plotting (action space is [-1, 1])
    x = np.linspace(-1.5, 1.5, 500)

    for i in range(7):
        ax = axes[i]
        mu, sigma = mean[i], std[i]

        # Gaussian PDF: (1 / (sigma * sqrt(2*pi))) * exp(-0.5 * ((x - mu) / sigma)^2)
        pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

        ax.plot(x, pdf, 'b-', linewidth=2)
        ax.fill_between(x, pdf, alpha=0.3)
        ax.axvline(mu, color='r', linestyle='--', label=f'μ={mu:.3f}')
        ax.axvline(mu - sigma, color='g', linestyle=':', alpha=0.7)
        ax.axvline(mu + sigma, color='g', linestyle=':', alpha=0.7, label=f'σ={sigma:.3f}')

        # Mark action bounds
        ax.axvline(-1, color='k', linestyle='-', alpha=0.3)
        ax.axvline(1, color='k', linestyle='-', alpha=0.3)

        ax.set_title(f'{action_names[i]}')
        ax.set_xlabel('Action value')
        ax.set_ylabel('Probability density')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.5, 1.5)

    # Hide the 8th subplot
    axes[7].axis('off')

    plt.suptitle('Action Distributions π(a|o) = N(μ, σ²)', fontsize=14)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"\nPlot saved to {save_path}")

    if show:
        plt.show()

    return fig


def print_action_info(model, obs, plot=False):
    """Print detailed action information for an observation."""
    mean, std = get_action_distribution(model, obs)
    action_det = get_action(model, obs, deterministic=True)
    action_stoch = get_action(model, obs, deterministic=False)

    print("\n" + "="*60)
    print("ACTION DISTRIBUTION")
    print("="*60)

    action_names = [
        "EE vel X", "EE vel Y", "EE vel Z",
        "EE ang vel (roll)", "EE ang vel (pitch)", "EE ang vel (yaw)",
        "Gripper"
    ]

    print(f"\n{'Dim':<5} {'Name':<20} {'Mean':>10} {'Std':>10} {'Det':>10} {'Sampled':>10}")
    print("-"*65)
    for i, name in enumerate(action_names):
        print(f"{i:<5} {name:<20} {mean[i]:>10.4f} {std[i]:>10.4f} {action_det[i]:>10.4f} {action_stoch[i]:>10.4f}")

    print("\nDet = Deterministic (mean), Sampled = Stochastic (random sample)")

    if plot:
        plot_action_distributions(mean, std)


def interactive_mode(model):
    """Interactive mode to query the policy."""
    obs_dim = model.observation_space.shape[0]

    print("\n" + "="*60)
    print("INTERACTIVE POLICY INSPECTOR")
    print("="*60)
    print(f"\nObservation dimension: {obs_dim}")
    print("\nCommands:")
    print("  'r' or 'random'  - Generate random observation")
    print("  'z' or 'zeros'   - Use zero observation")
    print("  'c' or 'custom'  - Enter custom observation values")
    print("  'p' or 'plot'    - Toggle plotting (current: OFF)")
    print("  'q' or 'quit'    - Exit")

    plot_enabled = False

    while True:
        print("\n" + "-"*40)
        cmd = input(f"Enter command (r/z/c/p/q) [plot={'ON' if plot_enabled else 'OFF'}]: ").strip().lower()

        if cmd in ['q', 'quit', 'exit']:
            print("Goodbye!")
            break

        elif cmd in ['p', 'plot']:
            plot_enabled = not plot_enabled
            print(f"Plotting {'ENABLED' if plot_enabled else 'DISABLED'}")

        elif cmd in ['r', 'random']:
            # Random observation within reasonable bounds
            obs = np.random.uniform(-1, 1, obs_dim).astype(np.float32)
            print(f"\nRandom observation (first 10 values): {obs[:10]}")
            print_action_info(model, obs, plot=plot_enabled)

        elif cmd in ['z', 'zeros']:
            obs = np.zeros(obs_dim, dtype=np.float32)
            print(f"\nZero observation")
            print_action_info(model, obs, plot=plot_enabled)

        elif cmd in ['c', 'custom']:
            print(f"\nEnter {obs_dim} comma-separated values (or press Enter for zeros):")
            try:
                values = input("> ").strip()
                if values == "":
                    obs = np.zeros(obs_dim, dtype=np.float32)
                else:
                    obs = np.array([float(x) for x in values.split(",")], dtype=np.float32)
                    if len(obs) != obs_dim:
                        print(f"Error: Expected {obs_dim} values, got {len(obs)}")
                        continue
                print_action_info(model, obs, plot=plot_enabled)
            except ValueError as e:
                print(f"Error parsing input: {e}")

        else:
            print("Unknown command. Use 'r', 'z', 'c', 'p', or 'q'")


def batch_query(model, observations):
    """Query policy with multiple observations at once."""
    results = []
    for obs in observations:
        mean, std = get_action_distribution(model, obs)
        action = get_action(model, obs, deterministic=True)
        results.append({
            'obs': obs,
            'mean': mean,
            'std': std,
            'action': action
        })
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect trained PPO policy")
    parser.add_argument("--model", type=str, default="trained_models/ppo_lift_500000_steps",
                        help="Path to trained model")
    parser.add_argument("--random", action="store_true",
                        help="Query with a random observation and exit")
    parser.add_argument("--zeros", action="store_true",
                        help="Query with zero observation and exit")
    parser.add_argument("--plot", action="store_true",
                        help="Show plot of action distributions")
    args = parser.parse_args()

    # Load model
    model = load_policy(args.model)

    if args.random:
        obs = np.random.uniform(-1, 1, model.observation_space.shape[0]).astype(np.float32)
        print_action_info(model, obs, plot=args.plot)
    elif args.zeros:
        obs = np.zeros(model.observation_space.shape[0], dtype=np.float32)
        print_action_info(model, obs, plot=args.plot)
    else:
        interactive_mode(model)
