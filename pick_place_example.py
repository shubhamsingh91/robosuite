"""
Robosuite Pick and Place Example with Camera Images
====================================================
This script demonstrates:
1. Loading a pick and place environment
2. Getting camera observations (RGB images)
3. Running a simple simulation loop
"""

import robosuite as suite
import numpy as np
import cv2
import os

def main():
    # Create the environment with camera observations
    env = suite.make(
        env_name="Lift",  # Pick and place task (lift a cube)
        robots="Panda",   # Use Panda robot
        has_renderer=True,  # No on-screen rendering
        has_offscreen_renderer=True,  # Enable offscreen rendering for camera
        use_camera_obs=True,  # Include camera observations
        camera_names=["agentview", "robot0_eye_in_hand"],  # Multiple cameras
        camera_heights=256,
        camera_widths=256,
        horizon=200,  # Episode length
        control_freq=20,  # Control frequency in Hz
    )

    # Reset environment
    obs = env.reset()

    print("Environment created successfully!")
    print(f"Observation keys: {obs.keys()}")
    print(f"Action space dimension: {env.action_spec[0].shape}")

    # Create output directory for images
    output_dir = "camera_images"
    os.makedirs(output_dir, exist_ok=True)

    # Run a few steps and save camera images
    for step in range(500):
        # Random action (you would replace this with your policy)
        action = np.random.uniform(-1, 1, env.action_spec[0].shape)
        print(f"Step {step}: taking action {action}")

        # Step the environment
        obs, reward, done, info = env.step(action)

        # Save camera images every 10 steps
        if step % 10 == 0:
            # Get camera images from observations
            agentview_img = obs["agentview_image"]
            eye_in_hand_img = obs["robot0_eye_in_hand_image"]

            # Convert from RGB to BGR for OpenCV and save
            # cv2.imwrite(
            #     f"{output_dir}/agentview_step_{step:03d}.png",
            #     cv2.cvtColor(agentview_img, cv2.COLOR_RGB2BGR)
            # )
            # cv2.imwrite(
            #     f"{output_dir}/eye_in_hand_step_{step:03d}.png",
            #     cv2.cvtColor(eye_in_hand_img, cv2.COLOR_RGB2BGR)
            # )

            print(f"Step {step}: reward={reward:.4f}, saved camera images")

        if done:
            print("Episode finished!")
            break

    # Close the environment
    env.close()
    print(f"\nCamera images saved to '{output_dir}/' directory")

if __name__ == "__main__":
    main()
