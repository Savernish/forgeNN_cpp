"""
Simple debug - print clearly
"""
import numpy as np
from rigidrl_py.envs.drone_env import DroneEnv

env = DroneEnv(render_mode=None)

def test_action(name, action, steps=30):
    obs, _ = env.reset()
    total_reward = 0
    for i in range(steps):
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        if term:
            break
    return i+1, total_reward

# Test different actions
print("Action           | Steps | Total Reward")
print("-" * 45)

steps, rew = test_action("No thrust     ", np.array([0.0, 0.0]))
print(f"No thrust        | {steps:5} | {rew:.1f}")

steps, rew = test_action("Full thrust   ", np.array([10.0, 10.0]))
print(f"Full thrust      | {steps:5} | {rew:.1f}")

steps, rew = test_action("Hover thrust  ", np.array([5.5, 5.5]))
print(f"Hover thrust     | {steps:5} | {rew:.1f}")

env.close()
