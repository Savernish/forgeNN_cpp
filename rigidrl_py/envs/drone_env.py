"""
DroneEnv - 2D drone hovering environment

A Gymnasium-compatible environment for training a drone to hover at a target position.
The drone has two motors (left/right) that can apply upward thrust.
"""

import numpy as np
from .base_env import RigidEnv
from .spaces import Box

# Import rigidRL
import sys
import os
core_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'diff_sim_core')
if os.path.exists(core_dir):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(core_dir)
    sys.path.insert(0, core_dir)

try:
    import rigidRL as rigid
except ImportError:
    rigid = None


# PARAMS FOR DRONE ENVIRONMENT 
# TODO: make these params configurable and probably not even be set in the environment itself,
#       rather in a child environment that the user creates / loads.

# Spawn points for the drone (randomly selected on reset)
SPAWN_POINTS = [
    (0.0, 1.5),    # Center only - simplify for now
]

# Target position (closer for easier learning)
TARGET_X = 0.0
TARGET_Y = 4.0  # Closer than 5.0

DRONE_MASS = 1.0
DRONE_WIDTH = 1.0
DRONE_HEIGHT = 0.2

MOTOR_LEFT_X = -0.4
MOTOR_LEFT_Y = 0.0
MOTOR_LEFT_WIDTH = 0.1
MOTOR_LEFT_HEIGHT = 0.1
MOTOR_LEFT_MASS = 0.05
MOTOR_LEFT_MAX_THRUST = 10.0

MOTOR_RIGHT_X = 0.4
MOTOR_RIGHT_Y = 0.0
MOTOR_RIGHT_WIDTH = 0.1
MOTOR_RIGHT_HEIGHT = 0.1
MOTOR_RIGHT_MASS = 0.05
MOTOR_RIGHT_MAX_THRUST = 10.0


class DroneEnv(RigidEnv):
    """
    Drone hovering environment.
    
    Task: Control a 2D drone with two motors to reach and maintain a target position.
    
    Observation Space (6D):
        [x, y, vx, vy, rotation, angular_velocity]
        - x, y: Position (meters)
        - vx, vy: Velocity (m/s)
        - rotation: Body angle (radians)
        - angular_velocity: Angular velocity (rad/s)
    
    Action Space (2D continuous):
        [left_thrust, right_thrust]
        - Each in range [0, max_thrust]
    
    Reward:
        - Negative distance to target
        - Velocity penalty
        - Angle penalty
        - Bonus for being close to target
        
    Episode ends when:
        - Terminated: Drone crashes (y < 0.1) or flips (|angle| > π/2)
        - Truncated: Max steps reached (default 500)
    """
    
    def __init__(
        self,
        render_mode=None,
        max_thrust: float = 10.0,
        target_x: float = TARGET_X,
        target_y: float = TARGET_Y,
        max_episode_steps: int = 500,
        **kwargs
    ):
        """
        Initialize drone environment.
        
        Args:
            render_mode: "human" for window, None for headless
            max_thrust: Maximum thrust per motor (N)
            target_x: Target X position (meters)
            target_y: Target Y position (meters)
            max_episode_steps: Steps before truncation
        """
        super().__init__(
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
            **kwargs
        )
        
        self.max_thrust = max_thrust
        self.target = np.array([target_x, target_y], dtype=np.float32)
        
        # Observation space: [dx_to_target, dy_to_target, vx, vy, rotation, angular_velocity]
        # Using relative target position so drone knows where to go from any spawn
        self.observation_space = Box(
            low=np.array([-10, -10, -10, -10, -np.pi, -10], dtype=np.float32),
            high=np.array([10, 10, 10, 10, np.pi, 10], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: [left_thrust, right_thrust]
        self.action_space = Box(
            low=np.array([0, 0], dtype=np.float32),
            high=np.array([max_thrust, max_thrust], dtype=np.float32),
            dtype=np.float32
        )
        
        # Drone components (set in _setup_scene)
        self.drone = None
        self.motor_left = None
        self.motor_right = None
        
    def _setup_scene(self):
        """Create the drone simulation scene."""
        # Set gravity
        self.engine.set_gravity(0, -9.81)
        
        # Ground plane
        self.engine.add_collider(0, -1, 20, 1, 0)
        
        # Randomly select spawn point
        spawn_idx = np.random.randint(0, len(SPAWN_POINTS))
        spawn_x, spawn_y = SPAWN_POINTS[spawn_idx]
        
        # Create drone body at spawn point
        self.drone = rigid.Body(spawn_x, spawn_y, DRONE_MASS, DRONE_WIDTH, DRONE_HEIGHT)
        
        # Add motors
        self.motor_left = rigid.Motor(MOTOR_LEFT_X, MOTOR_LEFT_Y, MOTOR_LEFT_WIDTH, MOTOR_LEFT_HEIGHT, MOTOR_LEFT_MASS, self.max_thrust)
        self.motor_right = rigid.Motor(MOTOR_RIGHT_X, MOTOR_RIGHT_Y, MOTOR_RIGHT_WIDTH, MOTOR_RIGHT_HEIGHT, MOTOR_RIGHT_MASS, self.max_thrust)
        
        self.drone.add_motor(self.motor_left)
        self.drone.add_motor(self.motor_right)
        
        self.engine.add_body(self.drone)
        
    def _get_obs(self) -> np.ndarray:
        """Get current observation with relative target position."""
        # Relative position to target (drone needs to know where to go)
        dx = self.target[0] - self.drone.get_x()
        dy = self.target[1] - self.drone.get_y()
        
        return np.array([
            dx,  # Relative X to target
            dy,  # Relative Y to target  
            self.drone.vel.get(0, 0),
            self.drone.vel.get(1, 0),
            self.drone.get_rotation(),
            self.drone.ang_vel.get(0, 0),
        ], dtype=np.float32)
        
    def _apply_action(self, action: np.ndarray):
        """Apply motor thrusts."""
        # Clip actions to valid range
        left_thrust = float(np.clip(action[0], 0, self.max_thrust))
        right_thrust = float(np.clip(action[1], 0, self.max_thrust))
        
        self.motor_left.thrust = left_thrust
        self.motor_right.thrust = right_thrust
        
    def _compute_reward(self) -> float:
        """Compute reward - simplified with strong gradients."""
        obs = self._get_obs()
        rel_pos = obs[:2]  # (dx, dy) to target
        vel = obs[2:4]
        angle = obs[4]
        
        # Distance to target
        dist = np.linalg.norm(rel_pos)
        
        # MAIN REWARD: exponential closeness (stronger gradient near target)
        # At dist=0: reward=1.0, at dist=5: reward=~0.007
        reward = np.exp(-dist)
        
        # Small angle penalty (don't flip)
        reward -= 0.1 * abs(angle)
        
        # Big bonus for being very close
        if dist < 0.5:
            reward += 1.0
        if dist < 0.2:
            reward += 2.0
            
        return float(reward)
        
    def _is_terminated(self) -> bool:
        """Check if episode should terminate."""
        y = self.drone.get_y()
        angle = self.drone.get_rotation()
        
        # Crashed into ground
        if y < 0.1:
            return True
            
        # Flipped over
        if abs(angle) > np.pi / 2:
            return True
            
        return False


# Simple test
if __name__ == "__main__":
    print("Testing DroneEnv...")
    
    # Test headless mode
    env = DroneEnv(render_mode=None)
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Initial obs: {obs}")
    
    # Take a few steps
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.3f}, pos=({obs[0]:.2f}, {obs[1]:.2f})")
        if term or trunc:
            break
            
    env.close()
    print("Test passed!")
