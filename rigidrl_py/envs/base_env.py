"""
RigidEnv - Base class for rigidRL environments

This class provides a Gymnasium-compatible interface for rigidRL physics simulations.
All concrete environments (DroneEnv, etc.) should inherit from this class.

Swap-point: Currently inherits from gymnasium.Env. Can be replaced with custom
base class later to remove gymnasium dependency.
"""

import gymnasium as gym
import numpy as np
from typing import Optional, Tuple, Dict, Any

# Import rigidRL - add path if needed
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
    print("Warning: rigidRL module not found. Run compile.bat first.")


class RigidEnv(gym.Env):
    """
    Base environment class for rigidRL simulations.
    
    Provides:
    - Gymnasium-compatible interface (reset, step, render, close)
    - Engine lifecycle management
    - Headless mode support for faster training
    
    Subclasses must implement:
    - _setup_scene(): Create bodies, colliders, etc.
    - _get_obs(): Return observation array
    - _compute_reward(): Return scalar reward
    - _is_terminated(): Return bool for episode end
    - observation_space: Define in __init__
    - action_space: Define in __init__
    """
    
    metadata = {
        "render_modes": ["human", "rgb_array", None],
        "render_fps": 60
    }
    
    def __init__(
        self,
        render_mode: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        scale: float = 50.0,
        dt: float = 0.016,
        substeps: int = 20,
        max_episode_steps: int = 500
    ):
        """
        Initialize base environment.
        
        Args:
            render_mode: "human" for window, None for headless
            width: Window width in pixels
            height: Window height in pixels  
            scale: Pixels per meter
            dt: Timestep in seconds
            substeps: Physics substeps per frame
            max_episode_steps: Maximum steps before truncation
        """
        super().__init__()
        
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.scale = scale
        self.dt = dt
        self.substeps = substeps
        self.max_episode_steps = max_episode_steps
        
        # Step counter
        self._step_count = 0
        
        # Engine instance (created in reset)
        self.engine = None
        
        # Subclasses must define these
        self.observation_space = None
        self.action_space = None
        
    def _create_engine(self):
        """Create the physics engine. Called during reset."""
        if rigid is None:
            raise RuntimeError("rigidRL module not available. Run compile.bat first.")
            
        headless = self.render_mode is None
        self.engine = rigid.Engine(
            self.width, 
            self.height, 
            self.scale, 
            self.dt, 
            self.substeps,
            headless
        )
        
    def _setup_scene(self):
        """
        Set up the simulation scene. Override in subclass.
        
        Should create bodies, add colliders, configure gravity, etc.
        """
        raise NotImplementedError("Subclass must implement _setup_scene()")
        
    def _get_obs(self) -> np.ndarray:
        """Return current observation. Override in subclass."""
        raise NotImplementedError("Subclass must implement _get_obs()")
        
    def _compute_reward(self) -> float:
        """Compute reward for current state. Override in subclass."""
        raise NotImplementedError("Subclass must implement _compute_reward()")
        
    def _is_terminated(self) -> bool:
        """Check if episode should terminate. Override in subclass."""
        raise NotImplementedError("Subclass must implement _is_terminated()")
        
    def _apply_action(self, action: np.ndarray):
        """Apply action to the simulation. Override in subclass."""
        raise NotImplementedError("Subclass must implement _apply_action()")
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional reset options (unused for now)
            
        Returns:
            observation: Initial observation
            info: Empty dict (can be extended)
        """
        super().reset(seed=seed)
        
        # Reset step counter
        self._step_count = 0
        
        # Create fresh engine
        if self.engine is not None:
            del self.engine
        self._create_engine()
        
        # Set up the scene (subclass implements)
        self._setup_scene()
        
        # Get initial observation
        obs = self._get_obs()
        info = {}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Take a step in the environment.
        
        Args:
            action: Action to apply
            
        Returns:
            observation: New observation
            reward: Scalar reward
            terminated: True if episode ended (goal/failure)
            truncated: True if episode truncated (time limit)
            info: Additional info dict
        """
        # Apply action
        self._apply_action(action)
        
        # Step physics
        if self.render_mode == "human":
            running = self.engine.step()
            if not running:
                # Window was closed
                return self._get_obs(), 0.0, True, False, {"window_closed": True}
        else:
            self.engine.update()
            
        self._step_count += 1
        
        # Get results
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self._step_count >= self.max_episode_steps
        info = {"step": self._step_count}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the environment.
        
        In "human" mode, rendering is handled by engine.step().
        In "rgb_array" mode, would return pixel array (not implemented yet).
        """
        if self.render_mode == "rgb_array":
            # TODO: Implement pixel capture from SDL
            return None
        # "human" mode is handled by engine.step()
        
    def close(self):
        """Clean up resources."""
        if self.engine is not None:
            del self.engine
            self.engine = None
