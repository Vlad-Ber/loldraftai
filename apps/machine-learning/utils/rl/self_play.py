# utils/rl/self_play.py

import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import gymnasium as gym
from gymnasium import Wrapper
from sb3_contrib import MaskablePPO


def action_mask_fn(env):
    """Action mask function that can be pickled."""
    return env.get_action_mask()


class ModelPool:
    """Maintains a pool of previous model iterations for self-play."""

    def __init__(
        self,
        save_dir: str,
        pool_size: int = 5,
        latest_model_prob: float = 0.5,
        random_opponent_prob: float = 0.1,
    ):
        self.save_dir = save_dir
        self.pool_size = pool_size
        self.latest_model_prob = latest_model_prob
        self.random_opponent_prob = random_opponent_prob
        self.model_pool: List[MaskablePPO] = []
        self.current_model: Optional[MaskablePPO] = None

    def update_pool(self, model: MaskablePPO, iteration: int):
        """Add new model to pool, potentially removing oldest if pool is full."""
        # Save current model
        model_path = os.path.join(self.save_dir, f"model_iter_{iteration}")
        model.save(model_path)

        # Load a copy for the pool to ensure independence
        pool_model = MaskablePPO.load(model_path)

        if len(self.model_pool) >= self.pool_size:
            self.model_pool.pop(0)  # Remove oldest model

        self.model_pool.append(pool_model)
        self.current_model = model

    def sample_opponent(self) -> Optional[MaskablePPO]:
        """Sample an opponent from the pool based on specified probabilities."""
        if random.random() < self.random_opponent_prob:
            return None  # Will use random opponent

        if not self.model_pool:
            return None

        if random.random() < self.latest_model_prob and self.current_model is not None:
            return self.current_model

        return random.choice(self.model_pool)


class SelfPlayWithPoolWrapper(Wrapper):
    """Environment wrapper that implements self-play with a pool of past iterations."""

    def __init__(
        self,
        env,
        model_pool: ModelPool,
        agent_side: str = "random",  # "blue", "red", or "random"
    ):
        super().__init__(env)
        self.model_pool = model_pool
        self.agent_side = agent_side
        self.current_side = None  # Will be set in reset()
        self.opponent_model = None

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        # Determine which side the learning agent plays
        self.current_side = (
            random.choice([0, 1])
            if self.agent_side == "random"
            else 0 if self.agent_side == "blue" else 1
        )

        # Sample opponent model for this episode
        self.opponent_model = self.model_pool.sample_opponent()

        # If it's opponent's turn first and we have a model, make their move
        if (
            self.env.current_step < len(self.env.draft_order)
            and self.env.draft_order[self.env.current_step]["team"] != self.current_side
        ):
            obs, reward, terminated, truncated, info = self.make_opponent_move(obs)

        return obs, info

    def step(self, action):
        # Agent's turn
        obs, reward, terminated, truncated, info = super().step(action)

        # Adjust reward based on agent's side
        if not terminated:
            reward = 0
        elif self.current_side == 1:  # If agent is red team
            reward = -reward  # Invert the reward

        # Make opponent moves until it's agent's turn again or episode ends
        while (
            not terminated
            and not truncated
            and self.env.current_step < len(self.env.draft_order)
            and self.env.draft_order[self.env.current_step]["team"] != self.current_side
        ):
            obs, reward, terminated, truncated, info = self.make_opponent_move(obs)

        return obs, reward, terminated, truncated, info

    def make_opponent_move(self, obs):
        """Make a move for the opponent (renamed from _make_opponent_move)."""
        if self.opponent_model is None:
            # Use random opponent logic
            action_info = self.env.get_action_info()
            action = self.get_opponent_action(action_info)
        else:
            # Use model to choose action
            action_masks = self.env.get_action_mask()
            action, _ = self.opponent_model.predict(
                obs, action_masks=action_masks, deterministic=False
            )

        return self.env.step(action)

    def get_opponent_action(self, action_info: Dict) -> int:
        """Get opponent's action (renamed from _get_opponent_action)."""
        phase = action_info["phase"]
        action_mask = self.env.get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]

        if phase == 0:  # Ban phase
            return self.np_random.choice(valid_actions)
        elif phase == 1:  # Pick phase
            return self.get_role_based_pick(valid_actions)

    def get_role_based_pick(self, valid_actions: List[int]) -> int:
        """Get role-based pick (renamed from _get_role_based_pick)."""
        # Get unpicked roles
        roles_picked = self.env.red_roles_picked  # opponent is always red team
        available_roles = [
            role for i, role in enumerate(self.env.roles) if roles_picked[i] == 0
        ]

        if not available_roles:
            return self.np_random.choice(valid_actions)

        # Choose a random available role
        chosen_role = self.np_random.choice(available_roles)

        # Get valid champions for this role that are also in valid actions
        role_champions = self.env.role_champion_sets[chosen_role]
        valid_role_champions = list(role_champions.intersection(valid_actions))

        if valid_role_champions:
            pick = self.np_random.choice(valid_role_champions)
        else:
            pick = self.np_random.choice(valid_actions)

        return pick
