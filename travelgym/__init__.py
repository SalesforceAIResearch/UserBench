"""
TravelGym: A Gymnasium environment for travel planning preference elicitation simulation.

This package provides a reinforcement learning environment where agents interact
with simulated users to elicit travel preferences for trip planning through
natural conversation and targeted questions.
"""

from .env import TravelEnv
from .config import (
    TravelGymConfig,
    get_default_config,
)

__version__ = "0.1.0"
__author__ = "Salesforce AI Research"
__email__ = "cqian@salesforce.com"

__all__ = [
    "TravelEnv",
    "TravelGymConfig",
    "get_default_config",
]

# Register the environment with Gymnasium
try:
    import gymnasium as gym
    gym.register(
        id='TravelGym-v0',
        entry_point='travelgym.env:TravelEnv',
        max_episode_steps=25,
    )
except ImportError:
    # Gymnasium not available, skip registration
    pass 