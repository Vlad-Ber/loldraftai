import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import List, Dict

from utils.rl import fetch_blue_side_winrate_prediction
from utils.match_prediction import MODEL_CONFIG_PATH
from utils.rl.champions import VALID_CHAMPION_IDS, ROLE_CHAMPIONS


def create_solo_queue_draft_order():
    # Define the draft order as a list of dicts
    draft_order = []
    # Ban phase: 5 bans per team
    for _ in range(5):
        draft_order.append({"team": 0, "phase": 0})  # Blue ban
        draft_order.append({"team": 1, "phase": 0})  # Red ban

    # Pick phase
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 1
    draft_order.append({"team": 1, "phase": 1})  # Red pick 1
    draft_order.append({"team": 1, "phase": 1})  # Red pick 2
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 2
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 3
    draft_order.append({"team": 1, "phase": 1})  # Red pick 3
    draft_order.append({"team": 1, "phase": 1})  # Red pick 4
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 4
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 5
    draft_order.append({"team": 1, "phase": 1})  # Red pick 5

    # Role selection phase: 5 picks per team
    for role_index in range(5):
        draft_order.append({"team": 0, "phase": 2, "role_index": role_index})
    for role_index in range(5):
        draft_order.append({"team": 1, "phase": 2, "role_index": role_index})

    return draft_order


class LoLDraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self):
        super(LoLDraftEnv, self).__init__()

        with open(MODEL_CONFIG_PATH, "rb") as f:
            model_params = pickle.load(f)

        self.num_champions = model_params["num_champions"]
        self.action_space = spaces.Discrete(self.num_champions)
        self.roles = ["TOP", "JUNGLE", "MID", "BOT", "UTILITY"]
        self.num_roles = len(self.roles)
        # Define the draft order for solo queue
        self.draft_order = create_solo_queue_draft_order()

        self.observation_space = spaces.Dict(
            {
                "available_champions": spaces.Box(
                    0, 1, shape=(self.num_champions,), dtype=np.int8
                ),
                "blue_picks": spaces.Box(
                    0, 1, shape=(5, self.num_champions), dtype=np.int8
                ),
                "red_picks": spaces.Box(
                    0, 1, shape=(5, self.num_champions), dtype=np.int8
                ),
                "blue_ordered_picks": spaces.Box(
                    0, 1, shape=(5, self.num_champions), dtype=np.int8
                ),
                "red_ordered_picks": spaces.Box(
                    0, 1, shape=(5, self.num_champions), dtype=np.int8
                ),
                "phase": spaces.Discrete(3),  # 0: ban, 1: pick, 2: role selection
                "turn": spaces.Discrete(2),  # 0: blue, 1: red
                "current_role": spaces.Discrete(
                    self.num_roles
                ),  # Role index during role selection
                "action_mask": spaces.Box(
                    0, 1, shape=(self.num_champions,), dtype=np.int8
                ),
            }
        )

        # Create a mask for valid champion IDs
        # This is can be set to restrict the model picks
        self.valid_champion_mask = np.zeros(self.num_champions, dtype=np.int8)
        self.valid_champion_mask[VALID_CHAMPION_IDS] = 1

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)  # Reset the RNG if seed is provided

        # Reset the draft state
        self.available_champions = np.ones(self.num_champions, dtype=np.int8)
        self.blue_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.red_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.blue_ordered_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.red_ordered_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.current_step = 0  # Index into self.draft_order
        self.done = False
        observation = self._get_obs()
        info = {}
        return observation, info

    def get_action_mask(self):
        action_info = self._get_action_info()
        team = action_info["team"]
        phase = action_info["phase"]

        if phase in [0, 1]:  # Ban or pick phase
            action_mask = self.available_champions.copy()
        elif phase == 2:  # Role selection phase
            picks, ordered_picks = self._get_team_picks(team)
            unassigned_champions = picks - ordered_picks
            action_mask = np.sum(unassigned_champions, axis=0)
        else:
            raise ValueError(f"Unknown phase: {phase}")
        return action_mask * self.valid_champion_mask

    def step(self, action):
        if self.done:
            raise Exception("Cannot call step() on a done environment")

        # Get current action info
        action_info = self.draft_order[self.current_step]
        current_team = action_info["team"]
        phase = action_info["phase"]
        current_role_index = action_info.get("role_index", None)

        # Process the action (pick or ban)
        valid = self._process_action(action, phase, current_team, current_role_index)
        if not valid:
            print(f"Invalid action: {action}")
            print(f"Available champions: {self.available_champions}")
            print(f"Blue picks: {self.blue_picks}")
            print(f"Red picks: {self.red_picks}")
            raise Exception("Invalid action")

        # Update the game state
        self._update_state()

        # Get the new observation
        observation = self._get_obs()

        # Check if the draft is complete
        terminated = self._is_draft_complete()
        truncated = False  # No truncation in this environment
        info = {}

        # Calculate the reward
        if terminated:
            reward = self._calculate_reward()
            self.done = True
        else:
            reward = 0

        return observation, reward, terminated, truncated, info

    def _process_action(self, action, phase, current_team, current_role_index):
        if phase in [0, 1] and self.available_champions[action] == 0:
            print(f"Champion {action} is not available")
            return False  # Invalid action

        if phase == 0:
            self._ban_champion(action)
        elif phase == 1:
            self._pick_champion(action, current_team)
        elif phase == 2:
            return self._assign_role(action, current_team, current_role_index)
        else:
            raise ValueError(f"Unknown phase: {phase}")

        return True

    def _ban_champion(self, action):
        self.available_champions[action] = 0

    def _pick_champion(self, action, current_team):
        self.available_champions[action] = 0
        picks, _ = self._get_team_picks(current_team)
        pick_index = np.where(np.sum(picks, axis=1) == 0)[0][0]
        picks[pick_index][action] = 1

    def _assign_role(self, action, current_team, current_role_index):
        picks, ordered_picks = self._get_team_picks(current_team)
        unassigned_champions = picks - ordered_picks
        if unassigned_champions[:, action].sum() == 0:
            print(
                f"Invalid action: Champion {action} is not in picks or already assigned"
            )
            return False  # Invalid action (champion not in picks or already assigned)
        ordered_picks[current_role_index][action] = 1
        return True

    def _get_team_picks(self, current_team):
        return (
            (self.blue_picks, self.blue_ordered_picks)
            if current_team == 0
            else (self.red_picks, self.red_ordered_picks)
        )

    def _update_state(self):
        self.current_step += 1
        if self.current_step >= len(self.draft_order):
            self.done = True

    def _get_action_info(self):
        # we allow a final observation
        return self.draft_order[min(self.current_step, len(self.draft_order) - 1)]

    def _get_obs(self):
        action_info = self._get_action_info()
        current_turn = action_info["team"]
        current_role_index = action_info.get(
            "role_index", 0
        )  # 0 if not in role selection, the model understands that can be ignored when phase is not 2

        if current_turn == 0:  # blue team
            # ennemy pick order is not visible
            blue_ordered_picks = self.blue_ordered_picks.copy()
            red_ordered_picks = np.zeros_like(self.red_ordered_picks)
        else:  # red team
            blue_ordered_picks = np.zeros_like(self.blue_ordered_picks)
            red_ordered_picks = self.red_ordered_picks.copy()

        phase = action_info["phase"]
        return {
            "available_champions": self.available_champions.copy(),
            "blue_picks": self.blue_picks.copy(),
            "red_picks": self.red_picks.copy(),
            "blue_ordered_picks": blue_ordered_picks,
            "red_ordered_picks": red_ordered_picks,
            "phase": np.array([phase], dtype=np.int8),
            "turn": np.array([current_turn], dtype=np.int8),
            "current_role": np.array([current_role_index], dtype=np.int8),
            "action_mask": self.get_action_mask(),
        }

    def _is_draft_complete(self):
        return self.current_step >= len(self.draft_order)

    def _calculate_reward(self):
        # Prepare input data for the external model

        # Extract the champion IDs from the ordered picks
        assert np.sum(self.blue_ordered_picks) == 5
        assert np.sum(self.red_ordered_picks) == 5
        blue_pick_ids = np.argmax(self.blue_ordered_picks, axis=1)
        red_pick_ids = np.argmax(self.red_ordered_picks, axis=1)

        # For simplicity, assume that the champion IDs are the indices
        champion_ids = np.concatenate([blue_pick_ids, red_pick_ids])

        winrate_prediction = fetch_blue_side_winrate_prediction(champion_ids)
        # Reward is winrate_prediction for blue team
        reward = winrate_prediction

        return reward

    def render(self):
        pass  # No rendering needed

    def close(self):
        pass  # Nothing to close


# Wrapper for self-play
class SelfPlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.opponent_picks = []

    def step(self, action):
        action_info = self.env._get_action_info()

        if action_info["team"] == 0:  # Agent's turn
            observation, reward, terminated, truncated, info = self.env.step(action)
        else:  # Opponent's turn
            opponent_action = self._get_opponent_action(action_info)
            observation, _, terminated, truncated, info = self.env.step(opponent_action)
            reward = 0

        while (
            not terminated
            and not truncated
            and self.env.current_step < len(self.env.draft_order)
            and self.env.draft_order[self.env.current_step]["team"] != 0
        ):
            action_info = self.env._get_action_info()
            opponent_action = self._get_opponent_action(action_info)
            observation, _, terminated, truncated, info = self.env.step(opponent_action)

        if terminated or truncated:
            reward = (
                self.env._calculate_reward()
                if self.env.current_step >= len(self.env.draft_order)
                else 0
            )

        return observation, reward, terminated, truncated, info

    def _get_opponent_action(self, action_info: Dict) -> int:
        phase = action_info["phase"]
        action_mask = self.env.get_action_mask()
        valid_actions = np.where(action_mask == 1)[0]

        if phase == 0:  # Ban phase
            return self.np_random.choice(valid_actions)
        elif phase == 1:  # Pick phase
            return self._get_role_based_pick(valid_actions)
        elif phase == 2:  # Role selection phase
            return self._get_role_selection(action_info["role_index"], valid_actions)

    def _get_role_based_pick(self, valid_actions: List[int]) -> int:
        current_role = len(self.opponent_picks)
        role_champions = set(ROLE_CHAMPIONS[current_role])
        valid_role_champions = list(role_champions.intersection(valid_actions))

        if valid_role_champions:
            pick = self.np_random.choice(valid_role_champions)
        else:
            pick = self.np_random.choice(valid_actions)

        self.opponent_picks.append(pick)
        return pick

    def _get_role_selection(self, role_index: int, valid_actions: List[int]) -> int:
        if role_index < len(self.opponent_picks):
            return self.opponent_picks[role_index]
        return self.np_random.choice(valid_actions)

    def reset(self, **kwargs):
        self.opponent_picks = []
        return super().reset(**kwargs)


def action_mask_fn(env):
    return env.get_action_mask()
