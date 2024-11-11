import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from itertools import permutations
from typing import List, Dict, TypedDict, Literal, Set

from utils.rl import fetch_blue_side_winrate_prediction, ROLE_CHAMPIONS_PATH
from utils.match_prediction import MODEL_CONFIG_PATH, ENCODERS_PATH
from utils.rl.champions import VALID_CHAMPION_IDS, ROLE_CHAMPIONS

RoleType = Literal["TOP", "JUNGLE", "MID", "BOT", "UTILITY"]

NOT_YET_PICKED = -1


class DraftStep(TypedDict, total=False):
    team: Literal[0, 1]  # 0 for blue, 1 for red
    phase: Literal[0, 1, 2]  # 0 for ban, 1 for pick, 2 for role selection
    role_index: int  # optional, only present in role selection phase


def create_solo_queue_draft_order() -> List[DraftStep]:
    # Define the draft order as a list of dicts
    draft_order: List[DraftStep] = []
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


def create_tournament_draft_order() -> List[DraftStep]:
    # Define the draft order as a list of dicts
    draft_order: List[DraftStep] = []
    # First Ban phase: 3 bans per team
    for _ in range(3):
        draft_order.append({"team": 0, "phase": 0})  # Blue ban
        draft_order.append({"team": 1, "phase": 0})  # Red ban

    # First Pick phase
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 1
    draft_order.append({"team": 1, "phase": 1})  # Red pick 1
    draft_order.append({"team": 1, "phase": 1})  # Red pick 2
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 2
    draft_order.append({"team": 0, "phase": 1})  # Blue pick 3
    draft_order.append({"team": 1, "phase": 1})  # Red pick 3

    # Second Ban Phase
    for _ in range(2):
        draft_order.append({"team": 1, "phase": 0})  # Red ban
        draft_order.append({"team": 0, "phase": 0})  # Blue ban

    # Second Pick Phase
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
        # This is required because ids are not consecutive, so some don't represent champions
        self.valid_champion_mask = np.zeros(self.num_champions, dtype=np.int8)
        self.valid_champion_mask[VALID_CHAMPION_IDS] = 1

        # Verify champion_to_role mapping has all valid champions
        for champ_id in VALID_CHAMPION_IDS:
            assert (
                champ_id in self.champion_to_role
            ), f"Champion {champ_id} has no role mapping!"

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
        action_info = self.get_current_draft_step()
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

    def step(self, action: int):
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

        self.current_step += 1
        if self.current_step >= len(self.draft_order):
            self.done = True

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

    def _process_action(
        self, action: int, phase: int, current_team: int, current_role_index: int
    ):
        # TODO: could also track the action history for visualizer
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

    def _ban_champion(self, action: int):
        self.available_champions[action] = 0

    def _pick_champion(self, action: int, current_team: int):
        self.available_champions[action] = 0
        picks, _ = self._get_team_picks(current_team)
        pick_index = np.where(np.sum(picks, axis=1) == 0)[0][0]
        picks[pick_index][action] = 1

    def _assign_role(self, action: int, current_team: int, current_role_index: int):
        picks, ordered_picks = self._get_team_picks(current_team)
        unassigned_champions = picks - ordered_picks
        if unassigned_champions[:, action].sum() == 0:
            print(
                f"Invalid action: Champion {action} is not in picks or already assigned"
            )
            return False  # Invalid action
        ordered_picks[current_role_index][action] = 1
        return True

    def _get_team_picks(self, current_team: int):
        return (
            (self.blue_picks, self.blue_ordered_picks)
            if current_team == 0
            else (self.red_picks, self.red_ordered_picks)
        )

    def get_current_draft_step(self) -> DraftStep:
        # we allow a final observation
        return self.draft_order[min(self.current_step, len(self.draft_order) - 1)]

    def _get_obs(self):
        action_info = self.get_current_draft_step()
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


def action_mask_fn(env: LoLDraftEnv):
    """Action mask function that can be pickled."""
    return env.get_action_mask()


class FixedRoleDraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, patches: List[int]):
        super(FixedRoleDraftEnv, self).__init__()

        with open(MODEL_CONFIG_PATH, "rb") as f:
            model_params = pickle.load(f)

        # Changes on reset:
        self.patches = patches
        self.current_patch = patches[-1]
        self.elos = [i for i in range(0, 4)]
        self.current_elo = self.elos[-1]

        # Fixed:
        self.num_champions = model_params["num_champions"]
        self.action_space = spaces.Discrete(self.num_champions)
        self.roles: List[RoleType] = ["TOP", "JUNGLE", "MID", "BOT", "UTILITY"]
        self.num_roles = len(self.roles)

        # Load role-champion mapping
        with open(ROLE_CHAMPIONS_PATH, "r") as f:
            self.role_champions: Dict[RoleType, List[int]] = json.load(f)

        # Convert champion IDs to sets for efficient lookup
        self.role_champion_sets = {
            role: set(champs) for role, champs in self.role_champions.items()
        }

        # Create champion to role mapping for quick lookups
        self.champion_to_role = {}
        for role, champions in self.role_champions.items():
            for champ in champions:
                self.champion_to_role[champ] = role

        # We can remove role selection phase from the draft order, because we are fixing the roles
        base_draft_order = create_tournament_draft_order()
        self.draft_order = [step for step in base_draft_order if step["phase"] != 2]

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
                "blue_roles_picked": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                "red_roles_picked": spaces.Box(0, 1, shape=(5,), dtype=np.int8),
                "phase": spaces.Discrete(2),  # 0: ban, 1: pick
                "turn": spaces.Discrete(2),  # 0: blue, 1: red
                "action_mask": spaces.Box(
                    0, 1, shape=(self.num_champions,), dtype=np.int8
                ),
                "numerical_patch": spaces.Discrete(len(self.patches) + 1),
                "numerical_elo": spaces.Discrete(len(self.elos) + 1),
            }
        )

        # Create valid champion mask
        self.valid_champion_mask = np.zeros(self.num_champions, dtype=np.int8)
        self.valid_champion_mask[VALID_CHAMPION_IDS] = 1

        self.reset()

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.current_patch = np.random.choice(self.patches)
        self.current_elo = np.random.choice(self.elos)

        self.available_champions = np.ones(self.num_champions, dtype=np.int8)
        self.blue_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.red_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        # Keep track of order for visualization
        self.blue_ordered_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.red_ordered_picks = np.zeros((5, self.num_champions), dtype=np.int8)
        self.blue_roles_picked = np.zeros(5, dtype=np.int8)
        self.red_roles_picked = np.zeros(5, dtype=np.int8)
        self.blue_pick_count = 0
        self.red_pick_count = 0
        self.current_step = 0
        self.done = False

        observation = self._get_obs()
        info = {}
        return observation, info

    def get_action_mask(self):
        action_info = self.get_current_draft_step()
        team = action_info["team"]
        phase = action_info["phase"]

        # Start with available champions, as they impact both ban and pick phases
        action_mask = self.available_champions.copy()

        if phase == 1:  # Pick phase
            # Get the roles that have already been picked
            roles_picked = (
                self.blue_roles_picked if team == 0 else self.red_roles_picked
            )
            picked_roles = {
                self.roles[i] for i in range(len(roles_picked)) if roles_picked[i] == 1
            }

            # Mask out champions from roles that have already been picked
            for champ_id in range(self.num_champions):
                if action_mask[champ_id] == 1:  # If champion is available
                    champ_role = self.champion_to_role.get(champ_id)
                    if champ_role in picked_roles:
                        action_mask[champ_id] = 0

        return action_mask * self.valid_champion_mask

    def step(self, action):
        if self.done:
            raise Exception("Cannot call step() on a done environment")

        action_info = self.draft_order[self.current_step]
        current_team = action_info["team"]
        phase = action_info["phase"]

        # Validate and process action
        valid = self._process_action(action, phase, current_team)
        if not valid:
            raise Exception("Invalid action")

        # Update state
        self.current_step += 1

        # Get new observation
        observation = self._get_obs()

        # Check if draft is complete
        terminated = self._is_draft_complete()
        truncated = False
        info = {}

        # Calculate reward
        if terminated:
            reward = self._calculate_reward()
            self.done = True
        else:
            reward = 0

        return observation, reward, terminated, truncated, info

    def _process_action(self, action, phase, current_team):
        if self.available_champions[action] == 0:
            return False

        if phase == 0:  # Ban phase
            self.available_champions[action] = 0
            return True

        elif phase == 1:  # Pick phase
            # Convert numpy array to integer before using as dictionary key
            action_int = action.item() if hasattr(action, "item") else int(action)
            # Get champion's role
            champ_role = self.champion_to_role.get(action_int)
            if champ_role is None:
                print("Invalid champion ID")
                return False

            # Get team's current roles and picks
            roles_picked = (
                self.blue_roles_picked if current_team == 0 else self.red_roles_picked
            )
            picks = self.blue_picks if current_team == 0 else self.red_picks
            ordered_picks = (
                self.blue_ordered_picks if current_team == 0 else self.red_ordered_picks
            )

            # Check if role is already picked
            role_index = self.roles.index(champ_role)
            if roles_picked[role_index] == 1:
                return False

            # Make the pick
            self.available_champions[action] = 0

            # Update role-based picks
            picks[role_index][action] = 1
            roles_picked[role_index] = 1

            # Update ordered picks for visualization
            if current_team == 0:
                ordered_picks[self.blue_pick_count][action] = 1
                self.blue_pick_count += 1
            else:
                ordered_picks[self.red_pick_count][action] = 1
                self.red_pick_count += 1

            return True

    def _is_draft_complete(self):
        is_complete = (
            np.sum(self.blue_roles_picked) == 5 and np.sum(self.red_roles_picked) == 5
        )

        if is_complete:
            # Create new ordered arrays in role order (TOP, JUNGLE, MID, BOT, UTILITY)
            new_blue_ordered = np.zeros_like(self.blue_ordered_picks)
            new_red_ordered = np.zeros_like(self.red_ordered_picks)

            # Fill the arrays in role order
            for role_idx, role in enumerate(self.roles):
                # Find champion for this role in blue team
                blue_champ = np.argmax(self.blue_picks[role_idx])
                new_blue_ordered[role_idx][blue_champ] = 1

                # Find champion for this role in red team
                red_champ = np.argmax(self.red_picks[role_idx])
                new_red_ordered[role_idx][red_champ] = 1

            # Update the ordered picks arrays
            self.blue_ordered_picks = new_blue_ordered
            self.red_ordered_picks = new_red_ordered

        return is_complete

    def _get_obs(self):
        action_info = self.get_current_draft_step()

        return {
            "available_champions": self.available_champions.copy(),
            "blue_picks": self.blue_picks.copy(),
            "red_picks": self.red_picks.copy(),
            "blue_ordered_picks": self.blue_ordered_picks.copy(),
            "red_ordered_picks": self.red_ordered_picks.copy(),
            "blue_roles_picked": self.blue_roles_picked.copy(),
            "red_roles_picked": self.red_roles_picked.copy(),
            "phase": np.array([action_info["phase"]], dtype=np.int8),
            "turn": np.array([action_info["team"]], dtype=np.int8),
            "action_mask": self.get_action_mask(),
            "numerical_patch": np.array([self.current_patch], dtype=np.int32),
            "numerical_elo": np.array([self.current_elo], dtype=np.int32),
        }

    def get_current_draft_step(self) -> DraftStep:
        return self.draft_order[min(self.current_step, len(self.draft_order) - 1)]

    def _calculate_reward(self):
        # Get picks in role order (TOP, JUNGLE, MID, BOT, UTILITY)
        champion_ids = []
        for role in self.roles:
            # Add blue team champion for this role
            for role_idx, r in enumerate(self.roles):
                if r == role:
                    champion_ids.append(np.argmax(self.blue_picks[role_idx]))
                    break

        for role in self.roles:
            # Add red team champion for this role
            for role_idx, r in enumerate(self.roles):
                if r == role:
                    champion_ids.append(np.argmax(self.red_picks[role_idx]))
                    break

        return fetch_blue_side_winrate_prediction(
            np.array(champion_ids),
            numerical_patch=int(self.current_patch),
            numerical_elo=int(self.current_elo),
        )


class DraftState:
    def __init__(self, num_champions: int, role_matrix: np.ndarray):
        self.blue_picks = []  # List of champion IDs
        self.red_picks = []  # List of champion IDs
        self.banned_champions = set()

        # Separate role matrices for each team
        self.blue_role_matrix = role_matrix.copy()
        self.red_role_matrix = role_matrix.copy()

        self.blue_roles = {}  # champion_id -> role
        self.red_roles = {}  # champion_id -> role
        self.blue_valid_combinations = []
        self.red_valid_combinations = []

    def copy(self) -> "DraftState":
        new_state = DraftState(len(self.blue_role_matrix), self.blue_role_matrix)
        new_state.blue_picks = self.blue_picks.copy()
        new_state.red_picks = self.red_picks.copy()
        new_state.banned_champions = self.banned_champions.copy()
        new_state.blue_role_matrix = self.blue_role_matrix.copy()
        new_state.red_role_matrix = self.red_role_matrix.copy()
        new_state.blue_roles = self.blue_roles.copy()
        new_state.red_roles = self.red_roles.copy()
        new_state.blue_valid_combinations = self.blue_valid_combinations.copy()
        new_state.red_valid_combinations = self.red_valid_combinations.copy()
        return new_state


class FlexibleRoleDraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, role_rates_path: str, min_playrate_threshold: float = 5):
        super().__init__()

        with open(MODEL_CONFIG_PATH, "rb") as f:
            model_params = pickle.load(f)

        self.num_champions = model_params["num_champions"]
        self.roles: List[RoleType] = ["TOP", "JUNGLE", "MID", "BOT", "UTILITY"]
        self.num_roles = len(self.roles)
        self.threshold = min_playrate_threshold

        # Load and decode role statistics
        self.role_rates = self._load_role_rates(role_rates_path)

        # Create role matrix
        self.original_role_matrix = self._create_role_matrix()

        # Action spaces
        self.action_space = spaces.Discrete(self.num_champions)

        # Observation space
        self.observation_space = spaces.Dict(
            {
                # TODO: can add back available champions but computed from role matrix
                "blue_picks": spaces.Box(
                    0, 1, shape=(5, self.num_champions), dtype=np.int8
                ),
                "red_picks": spaces.Box(
                    0, 1, shape=(5, self.num_champions), dtype=np.int8
                ),
                "blue_role_matrix": spaces.Box(
                    0, 1, shape=(self.num_champions, self.num_roles), dtype=np.int8
                ),
                "red_role_matrix": spaces.Box(
                    0, 1, shape=(self.num_champions, self.num_roles), dtype=np.int8
                ),
                "blue_roles": spaces.Box(
                    0, 1, shape=(5, self.num_roles), dtype=np.int8
                ),
                "red_roles": spaces.Box(0, 1, shape=(5, self.num_roles), dtype=np.int8),
                "phase": spaces.Discrete(3),  # 0: ban, 1: pick, 2: role
                "turn": spaces.Discrete(2),  # 0: blue, 1: red
                "action_mask": spaces.Box(
                    0, 1, shape=(self.num_champions,), dtype=np.int8
                ),
            }
        )

        # Create valid champion mask with original IDs
        self.valid_champion_mask = np.zeros(self.num_champions, dtype=np.int8)
        self.valid_champion_mask[VALID_CHAMPION_IDS] = 1

        # Validate that all valid champion IDs are within bounds
        if max(VALID_CHAMPION_IDS) >= self.num_champions:
            raise ValueError(
                f"Valid champion IDs must be less than num_champions ({self.num_champions})"
            )

        # Draft order from base implementation
        self.draft_order = create_solo_queue_draft_order()

    def _load_role_rates(self, path: str) -> Dict[int, Dict[str, float]]:
        """Load role statistics that are already stored with original champion IDs."""
        with open(path, "r") as f:
            role_rates = json.load(f)

        # Just convert string keys to integers
        return {int(champ_id): roles for champ_id, roles in role_rates.items()}

    def _create_role_matrix(self) -> np.ndarray:
        """Create binary matrix of valid roles per champion based on threshold."""
        total_valid_role_champion_pairs = 0
        matrix = np.zeros((self.num_champions, self.num_roles), dtype=np.int8)

        # Use decoded_role_rates
        for champ_id in self.role_rates:
            rates = self.role_rates[champ_id]
            for role_idx, role in enumerate(self.roles):
                # Convert comparison result to int8
                if np.float32(rates.get(role, 0.0)) >= np.float32(self.threshold):
                    matrix[champ_id, role_idx] = 1
                    total_valid_role_champion_pairs += 1
        # print(f"Total valid role-champion pairs: {total_valid_role_champion_pairs}")
        return matrix

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize draft state
        self.state = DraftState(self.num_champions, self.original_role_matrix)
        self.current_step = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def get_current_draft_step(self) -> DraftStep:
        return self.draft_order[min(self.current_step, len(self.draft_order) - 1)]

    def _get_valid_role_combinations(
        self, picked_champions: List[int], team: int
    ) -> List[Dict[int, RoleType]]:
        """
        Find all valid role combinations for picked champions plus NOT_YET_PICKED placeholders.
        Each combination is a mapping of champion IDs (or NOT_YET_PICKED) to roles.
        """
        num_picks = len(picked_champions)
        roles = self.roles.copy()
        role_indices = list(range(self.num_roles))

        # Get the current role matrix for determining available roles for NOT_YET_PICKED
        current_role_matrix = (
            self.state.blue_role_matrix if team == 0 else self.state.red_role_matrix
        )

        # Use original matrix for picked champions' viable roles
        champ_role_matrix = (
            self.original_role_matrix[picked_champions, :]
            if num_picks > 0
            else np.array([])
        )

        # Build a list of viable roles for each picked champion
        champ_viable_roles = []
        for i in range(num_picks):
            viable_roles = [
                roles[j] for j in range(self.num_roles) if champ_role_matrix[i, j]
            ]
            champ_viable_roles.append(viable_roles)

        # Add NOT_YET_PICKED champions to fill up to 5 positions
        remaining_picks = 5 - num_picks
        for _ in range(remaining_picks):
            # NOT_YET_PICKED can go to any role that has at least one valid champion
            viable_roles = [
                roles[j]
                for j in range(self.num_roles)
                if np.any(
                    current_role_matrix[:, j]
                )  # At least one champion can still play this role
            ]
            champ_viable_roles.append(viable_roles)

        valid_combinations = []

        # Generate all permutations of roles (role assignments)
        # Now we're always working with 5 positions (picked + NOT_YET_PICKED)
        for role_indices_perm in permutations(role_indices, 5):
            # Map the permuted role indices to roles
            assigned_roles = [roles[idx] for idx in role_indices_perm]

            # For each position (picked champion or NOT_YET_PICKED), check if assigned role is viable
            valid = True
            for i in range(5):
                assigned_role = assigned_roles[i]
                if assigned_role not in champ_viable_roles[i]:
                    valid = False
                    break

            if valid:
                # Build the assignment dictionary
                assignment = {}
                for i in range(num_picks):
                    assignment[picked_champions[i]] = assigned_roles[i]
                for i in range(num_picks, 5):
                    assignment[NOT_YET_PICKED] = assigned_roles[i]
                valid_combinations.append(assignment)

        return valid_combinations

    def _initialize_role_selection(self, team: int):
        """
        Generate all valid role combinations for a team when entering role selection phase
        """
        picks = self.state.blue_picks if team == 0 else self.state.red_picks
        combinations = self._get_valid_role_combinations(picks)

        if team == 0:
            self.state.blue_valid_combinations = combinations
        else:
            self.state.red_valid_combinations = combinations

        print(f"Generated {len(combinations)} valid combinations for team {team}")

    def _update_role_availability(self) -> None:
        """
        After each pick, identify locked roles and update role matrices accordingly.
        Each team has its own role matrix and locked roles.
        A role is locked (zeroed out) if it can never be assigned to NOT_YET_PICKED
        in any valid combination.
        """
        # Process blue team
        if self.state.blue_picks:
            valid_combinations = self._get_valid_role_combinations(
                self.state.blue_picks, 0
            )
            print(
                f"Valid combinations for blue team: {valid_combinations}, blue_picks: {self.state.blue_picks}"
            )
            # For each role, check if it's always assigned to a picked champion
            for role_idx, role in enumerate(self.roles):
                # For each combination, check what champion is assigned to this role
                role_assignments = []
                for combo in valid_combinations:
                    # Find which champion is assigned to this role
                    for champ, assigned_role in combo.items():
                        if assigned_role == role:
                            role_assignments.append(champ)
                            break

                # If this role is never assigned to NOT_YET_PICKED, it must be locked
                if NOT_YET_PICKED not in role_assignments:
                    # Zero out this role for all champions
                    for champ_id in range(len(self.state.blue_role_matrix)):
                        self.state.blue_role_matrix[champ_id, role_idx] = 0

        # Process red team
        if self.state.red_picks:
            valid_combinations = self._get_valid_role_combinations(
                self.state.red_picks, 1
            )
            # For each role, check if it's always assigned to a picked champion
            for role_idx, role in enumerate(self.roles):
                # For each combination, check what champion is assigned to this role
                role_assignments = []
                for combo in valid_combinations:
                    # Find which champion is assigned to this role
                    for champ, assigned_role in combo.items():
                        if assigned_role == role:
                            role_assignments.append(champ)
                            break

                # If this role is never assigned to NOT_YET_PICKED, it must be locked
                if NOT_YET_PICKED not in role_assignments:
                    # Zero out this role for all champions
                    for champ_id in range(len(self.state.red_role_matrix)):
                        self.state.red_role_matrix[champ_id, role_idx] = 0

    def get_action_mask(self) -> np.ndarray:
        """Generate action mask based on current state and phase"""
        if self.current_step >= len(self.draft_order):
            # this can happen when doing a last observation, because action mask is in obs
            return np.zeros(self.num_champions, dtype=np.int8)

        action_info = self.draft_order[self.current_step]
        team = action_info["team"]
        phase = action_info["phase"]

        mask = np.zeros(self.num_champions, dtype=np.int8)
        role_matrix = (
            self.state.blue_role_matrix if team == 0 else self.state.red_role_matrix
        )

        if phase == 0:  # Ban phase
            mask = (
                np.any(self.state.blue_role_matrix, axis=1)
                | np.any(self.state.red_role_matrix, axis=1)
            ).astype(np.int8)

        elif phase == 1:  # Pick phase
            # Only check role matrix for current team to determine valid picks
            mask = np.any(role_matrix, axis=1).astype(np.int8)

        elif phase == 2:  # Role selection phase
            role_index = action_info["role_index"]
            combinations = (
                self.state.blue_valid_combinations
                if team == 0
                else self.state.red_valid_combinations
            )

            valid_champions = set()
            for combo in combinations:
                for champ, role in combo.items():
                    if self.roles.index(role) == role_index:
                        valid_champions.add(champ)

            for champ in valid_champions:
                mask[champ] = 1

            # Add debug print to see valid champions
            print(
                f"Phase {phase}, team {team}, step {self.current_step}, valid champions for role {self.roles[role_index]}: {valid_champions}"
            )

        action_mask = mask * self.valid_champion_mask
        print(
            f"Phase {phase}, team {team}, step {self.current_step}, non zero values count: {np.count_nonzero(action_mask)}"
        )

        return action_mask

    def step(self, action: int):
        if self.current_step >= len(self.draft_order):
            raise Exception("Draft is already complete")

        action_info = self.draft_order[self.current_step]
        team = action_info["team"]
        phase = action_info["phase"]

        # Process action
        valid = self._process_action(action, phase, team)
        if not valid:
            print(f"Invalid action: {action} for phase {phase} and team {team}")
            raise Exception("Invalid action")

        # Update state
        self.current_step += 1

        # Get new observation
        observation = self._get_obs()

        # Check if draft is complete
        terminated = self.current_step >= len(self.draft_order)
        truncated = False

        # Calculate reward
        reward = self._calculate_reward() if terminated else 0

        info = {}
        return observation, reward, terminated, truncated, info

    def _process_action(self, action: int, phase: int, team: int) -> bool:
        """Process action based on current phase"""
        if phase == 0:  # Ban phase
            if action in self.state.banned_champions:
                return False
            self.state.banned_champions.add(action)
            # Zero out all roles for this champion in both matrices
            self.state.blue_role_matrix[action, :] = 0
            self.state.red_role_matrix[action, :] = 0

        elif phase == 1:  # Pick phase
            picked_champs = set(self.state.blue_picks + self.state.red_picks)
            if action in picked_champs or action in self.state.banned_champions:
                return False

            # Zero out all roles for this champion in both matrices
            self.state.blue_role_matrix[action, :] = 0
            self.state.red_role_matrix[action, :] = 0

            if team == 0:
                self.state.blue_picks.append(action)
            else:
                self.state.red_picks.append(action)

            self._update_role_availability()

            # Generate valid combinations for completed teams
            if (
                len(self.state.blue_picks) == 5
                and not self.state.blue_valid_combinations
            ):
                self.state.blue_valid_combinations = self._get_valid_role_combinations(
                    self.state.blue_picks, 0
                )
            if len(self.state.red_picks) == 5 and not self.state.red_valid_combinations:
                self.state.red_valid_combinations = self._get_valid_role_combinations(
                    self.state.red_picks, 1
                )

        elif phase == 2:  # Role selection phase
            role_index = self.draft_order[self.current_step]["role_index"]
            current_role = self.roles[role_index]
            combinations = (
                self.state.blue_valid_combinations
                if team == 0
                else self.state.red_valid_combinations
            )

            # Filter combinations to those where this champion plays this role
            new_combinations = [
                combo for combo in combinations if combo.get(action) == current_role
            ]
            print(f"Old combinations: {combinations}")
            print(f"New combinations: {new_combinations}")
            print(f"Action: {action}, Role: {current_role}")

            if not new_combinations:
                return False

            # Update valid combinations
            if team == 0:
                self.state.blue_valid_combinations = new_combinations
                self.state.blue_roles[action] = current_role
            else:
                self.state.red_valid_combinations = new_combinations
                self.state.red_roles[action] = current_role

        return True

    def _get_obs(self):
        """Convert current state to observation"""
        action_info = self.draft_order[
            min(self.current_step, len(self.draft_order) - 1)
        ]

        # Convert picks to one-hot representation
        blue_picks_onehot = np.zeros((5, self.num_champions), dtype=np.int8)
        red_picks_onehot = np.zeros((5, self.num_champions), dtype=np.int8)

        for i, champ in enumerate(self.state.blue_picks[:5]):
            blue_picks_onehot[i, champ] = 1
        for i, champ in enumerate(self.state.red_picks[:5]):
            red_picks_onehot[i, champ] = 1

        # Convert role assignments to matrices
        blue_roles = np.zeros((5, self.num_roles), dtype=np.int8)
        red_roles = np.zeros((5, self.num_roles), dtype=np.int8)

        for champ_id, role in self.state.blue_roles.items():
            idx = self.state.blue_picks.index(champ_id)
            role_idx = self.roles.index(role)
            blue_roles[idx, role_idx] = 1

        for champ_id, role in self.state.red_roles.items():
            idx = self.state.red_picks.index(champ_id)
            role_idx = self.roles.index(role)
            red_roles[idx, role_idx] = 1

        return {
            "blue_picks": blue_picks_onehot,
            "red_picks": red_picks_onehot,
            "blue_role_matrix": self.state.blue_role_matrix.copy(),
            "red_role_matrix": self.state.red_role_matrix.copy(),
            "blue_roles": blue_roles,
            "red_roles": red_roles,
            "phase": np.array([action_info["phase"]], dtype=np.int8),
            "turn": np.array([action_info["team"]], dtype=np.int8),
            "action_mask": self.get_action_mask(),
        }

    def _calculate_reward(self) -> float:
        """Calculate final reward based on team compositions"""
        # Verify all roles are assigned
        if len(self.state.blue_roles) != 5 or len(self.state.red_roles) != 5:
            raise Exception("Not all roles are assigned")

        # Create ordered champion list (TOP, JUNGLE, MID, BOT, UTILITY for both teams)
        champion_ids = []

        # Add blue team champions in role order
        for role in self.roles:
            champ = next(
                champ_id
                for champ_id, assigned_role in self.state.blue_roles.items()
                if assigned_role == role
            )
            champion_ids.append(champ)

        # Add red team champions in role order
        for role in self.roles:
            champ = next(
                champ_id
                for champ_id, assigned_role in self.state.red_roles.items()
                if assigned_role == role
            )
            champion_ids.append(champ)

        # Use external model for win rate prediction
        return np.float32(
            fetch_blue_side_winrate_prediction(np.array(champion_ids, dtype=np.int32))
        )
