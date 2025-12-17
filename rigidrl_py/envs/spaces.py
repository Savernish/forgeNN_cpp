# Abstraction layer over gymnasium.spaces
# Swap-point: Replace with custom implementations later to remove gymnasium dependency

from gymnasium.spaces import Space, Box, Discrete, Dict, Tuple, MultiBinary, MultiDiscrete

__all__ = ['Space', 'Box', 'Discrete', 'Dict', 'Tuple', 'MultiBinary', 'MultiDiscrete']
