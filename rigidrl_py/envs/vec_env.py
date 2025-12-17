"""
Vectorized environment wrappers for rigidRL.

Uses Gymnasium's built-in vectorization for parallel environment execution.
This is a swap-point: can be replaced with custom vectorization later.
"""

from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from typing import Optional, Callable, List


def make_vec_env(
    env_fn: Callable,
    num_envs: int = 4,
    async_envs: bool = False,
) -> SyncVectorEnv:
    """
    Create a vectorized environment for parallel training.
    
    Args:
        env_fn: Factory function that creates a single environment instance
        num_envs: Number of parallel environments
        async_envs: If True, use AsyncVectorEnv (multiprocessing)
                   If False, use SyncVectorEnv (sequential in single process)
                   
    Returns:
        Vectorized environment
        
    Example:
        >>> from rigidrl_py.envs import DroneEnv, make_vec_env
        >>> vec_env = make_vec_env(lambda: DroneEnv(), num_envs=8)
        >>> obs, info = vec_env.reset()
        >>> print(obs.shape)  # (8, 6)
    """
    env_fns = [env_fn for _ in range(num_envs)]
    
    if async_envs:
        return AsyncVectorEnv(env_fns)
    else:
        return SyncVectorEnv(env_fns)


def make_drone_vec_env(
    num_envs: int = 4,
    async_envs: bool = False,
    **kwargs
) -> SyncVectorEnv:
    """
    Create vectorized DroneEnv instances.
    
    This is a convenience function specifically for drone training.
    All environments are created in headless mode (no rendering).
    
    Args:
        num_envs: Number of parallel environments
        async_envs: Use multiprocessing if True
        **kwargs: Additional arguments passed to DroneEnv
        
    Returns:
        Vectorized DroneEnv
        
    Example:
        >>> from rigidrl_py.envs import make_drone_vec_env
        >>> vec_env = make_drone_vec_env(num_envs=8, max_thrust=15.0)
        >>> obs, _ = vec_env.reset()
        >>> actions = vec_env.action_space.sample()  # (8, 2)
        >>> obs, rewards, terms, truncs, infos = vec_env.step(actions)
    """
    from .drone_env import DroneEnv
    
    def make_env():
        return DroneEnv(render_mode=None, **kwargs)
    
    return make_vec_env(make_env, num_envs=num_envs, async_envs=async_envs)
