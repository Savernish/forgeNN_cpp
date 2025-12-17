"""
rigidRL Environment Module

Gymnasium-compatible RL environments for rigidRL physics engine.
"""

from .base_env import RigidEnv
from .spaces import Space, Box, Discrete
from .drone_env import DroneEnv
from .vec_env import make_vec_env, make_drone_vec_env

__all__ = [
    'RigidEnv', 
    'DroneEnv',
    'Space', 'Box', 'Discrete',
    'make_vec_env', 'make_drone_vec_env'
]
