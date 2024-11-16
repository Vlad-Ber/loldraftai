import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import json
from itertools import permutations
from typing import List, Dict, TypedDict, Literal, Set
import random

from utils.rl import (
    fetch_blue_side_winrate_prediction,
    ROLE_CHAMPIONS_PATH,
    ROLE_PLAYRATES_PATH,
)
from utils.match_prediction import MODEL_CONFIG_PATH, ENCODERS_PATH
from utils.rl.champions import VALID_CHAMPION_IDS, ROLE_CHAMPIONS

RoleType = Literal["TOP", "JUNGLE", "MID", "BOT", "UTILITY"]

NOT_YET_PICKED = -1

DEBUG_PRINT = False


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


class DraftState:
    def __init__(self, num_champions: int, role_matrix: np.ndarray):
        self.blue_picks = []  # List of champion IDs
        self.red_picks = []  # List of champion IDs
        self.banned_champions = set()
        self.num_roles = 5

        # Separate role matrices for each team
        self.blue_role_matrix = self._apply_random_masking(role_matrix.copy())
        self.red_role_matrix = self._apply_random_masking(role_matrix.copy())

        self.blue_fallback_role_matrix = role_matrix.copy()
        self.blue_fallbacks_used = 0
        self.red_fallback_role_matrix = role_matrix.copy()
        self.red_fallbacks_used = 0

        self.blue_roles = {}  # champion_id -> role
        self.red_roles = {}  # champion_id -> role
        self.blue_valid_combinations = []
        self.red_valid_combinations = []

        # set at champion pick, to take into account the matrix at the time of pick
        self.blue_pick_viable_roles: Dict[int, List[RoleType]] = (
            {}
        )  # champion_id -> list of viable roles
        self.red_pick_viable_roles: Dict[int, List[RoleType]] = {}

    def _apply_random_masking(self, role_matrix: np.ndarray) -> np.ndarray:
        """
        Randomly mask (set to 0) some valid champion-role pairs in the matrix.
        Guarantees at least one valid champion per role after masking.

        Args:
            role_matrix: Original role matrix with shape (num_champions, num_roles)

        Returns:
            Masked role matrix with same shape
        """
        # Add type hint for clarity
        masked_matrix: np.ndarray = role_matrix.copy()

        # Consider adding these as class attributes or parameters
        NO_MASK_PROBABILITY = 0.1
        MASK_MEAN_RATIO = 0.5
        MASK_STD_RATIO = 0.2

        if random.random() < NO_MASK_PROBABILITY:
            return masked_matrix

        # For each role, randomly mask some valid champions
        for role_idx in range(self.num_roles):
            valid_champions = np.where(role_matrix[:, role_idx] == 1)[0]

            if len(valid_champions) == 0:
                continue

            # Determine number of champions to mask using normal distribution
            # Mean: 50% of valid champions, Std: 20% of valid champions
            # Clamp between 0 and len(valid_champions)-1 to ensure at least one remains
            num_to_mask = int(
                np.random.normal(
                    loc=len(valid_champions) * MASK_MEAN_RATIO,
                    scale=len(valid_champions) * MASK_STD_RATIO,
                )
            )
            num_to_mask = max(0, min(num_to_mask, len(valid_champions) - 1))

            # Randomly select champions to mask
            champions_to_mask = np.random.choice(
                valid_champions, size=num_to_mask, replace=False
            )

            # Apply masking
            masked_matrix[champions_to_mask, role_idx] = 0

        # Verify and fix roles with no valid champions
        for role_idx in range(self.num_roles):
            if not np.any(masked_matrix[:, role_idx]):
                # should not happen because we clamp num_to_mask to be at least 1
                print(f"No champions available for role {self.roles[role_idx]}")
                # If no champions available for this role, randomly add one from original matrix
                valid_champions = np.where(role_matrix[:, role_idx] == 1)[0]
                if len(valid_champions) > 0:
                    champion_to_add = np.random.choice(valid_champions)
                    masked_matrix[champion_to_add, role_idx] = 1

        return masked_matrix


class FlexibleRoleDraftEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        # default to global playrates with at least 0.5% global presence
        patches: List[int],
        role_rates_path: str = ROLE_PLAYRATES_PATH,
        min_playrate_threshold: float = 0.5,
    ):
        super().__init__()

        with open(MODEL_CONFIG_PATH, "rb") as f:
            model_params = pickle.load(f)

        self.num_champions = model_params["num_champions"]
        self.roles: List[RoleType] = ["TOP", "JUNGLE", "MID", "BOT", "UTILITY"]
        self.num_roles = len(self.roles)
        self.threshold = min_playrate_threshold
        self.patches = patches
        self.patch = random.choice(self.patches)

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
                "numerical_patch": spaces.Box(
                    low=float(min(self.patches)),
                    high=float(max(self.patches)),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        # Create valid champion mask with original IDs
        self.valid_champion_mask = np.zeros(self.num_champions, dtype=np.int8)
        # TODO: should be per patch, or actually, we can remove because we use playrate matrix
        self.valid_champion_mask[VALID_CHAMPION_IDS] = 1

        # Validate that all valid champion IDs are within bounds
        if max(VALID_CHAMPION_IDS) >= self.num_champions:
            raise ValueError(
                f"Valid champion IDs must be less than num_champions ({self.num_champions})"
            )

        # Draft order from base implementation
        self.draft_order = create_tournament_draft_order()

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
        return matrix

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Initialize draft state
        self.state = DraftState(self.num_champions, self.original_role_matrix.copy())
        self.current_step = 0
        self.patch = random.choice(self.patches)

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

        # Get viable roles dictionary for the appropriate team
        pick_viable_roles = (
            self.state.blue_pick_viable_roles
            if team == 0
            else self.state.red_pick_viable_roles
        )

        # Build a list of viable roles for each picked champion
        champ_viable_roles = []
        for champ_id in picked_champions:
            viable_roles = pick_viable_roles[champ_id]
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

        if DEBUG_PRINT:
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
            if len(valid_combinations) == 0:
                raise Exception("No valid combinations for blue team")
            if DEBUG_PRINT:
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
                        self.state.blue_fallback_role_matrix[champ_id, role_idx] = 0

        # Process red team
        if self.state.red_picks:
            valid_combinations = self._get_valid_role_combinations(
                self.state.red_picks, 1
            )
            if len(valid_combinations) == 0:
                raise Exception("No valid combinations for red team")
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
                        self.state.red_fallback_role_matrix[champ_id, role_idx] = 0

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
            if DEBUG_PRINT:
                print(
                    f"Phase {phase}, team {team}, step {self.current_step}, valid champions for role {self.roles[role_index]}: {valid_champions}"
                )

        # should not be needed anymore because we use playrate matrix?
        # action_mask = mask * self.valid_champion_mask
        action_mask = mask
        if DEBUG_PRINT:
            print(
                f"Phase {phase}, team {team}, step {self.current_step}, non zero values count: {np.count_nonzero(action_mask)}"
            )
            if np.count_nonzero(action_mask) == 0:
                print(f"No valid champions for team {team} at step {self.current_step}")

        return action_mask

    def step(self, action: int):
        if self.current_step >= len(self.draft_order):
            raise Exception("Draft is already complete")

        # Convert numpy array to integer at the entry point
        # Sometimes can be changed to numpy through multiple wrappers
        # TODO: debug why exactly, it happened in visualisation notebook
        action = action.item() if hasattr(action, "item") else int(action)

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
            # Zero out all roles for this champion for both teams
            self.state.blue_role_matrix[action, :] = 0
            self.state.blue_fallback_role_matrix[action, :] = 0
            self.state.red_role_matrix[action, :] = 0
            self.state.red_fallback_role_matrix[action, :] = 0
            self._apply_fallbacks()

        elif phase == 1:  # Pick phase
            picked_champs = set(self.state.blue_picks + self.state.red_picks)
            if action in picked_champs or action in self.state.banned_champions:
                return False

            # Store what roles this champion has at the time of picking
            # Get current role matrix for the team
            role_matrix = (
                self.state.blue_role_matrix if team == 0 else self.state.red_role_matrix
            )
            # Store viable roles for this champion at time of picking
            viable_roles = [
                self.roles[j] for j in range(self.num_roles) if role_matrix[action, j]
            ]
            if not viable_roles:
                if DEBUG_PRINT:
                    print(
                        f"Warning: Champion {action} has no viable roles at time of picking"
                    )
                return False
            # Store viable roles in appropriate team's dictionary
            if team == 0:
                self.state.blue_pick_viable_roles[action] = viable_roles
                self.state.blue_picks.append(action)
            else:
                self.state.red_pick_viable_roles[action] = viable_roles
                self.state.red_picks.append(action)

            # Zero out all roles for this champion for both teams
            self.state.blue_role_matrix[action, :] = 0
            self.state.blue_fallback_role_matrix[action, :] = 0
            self.state.red_role_matrix[action, :] = 0
            self.state.red_fallback_role_matrix[action, :] = 0
            self._apply_fallbacks()

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
            if DEBUG_PRINT:
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

    def _apply_fallbacks(self):
        """Apply fallback matrices if needed."""
        # Use fallback matrices if needed.
        for role_idx in range(self.num_roles):
            # Check blue team's role availability
            blue_role_available = np.any(self.state.blue_role_matrix[:, role_idx])
            blue_fallback_available = np.any(
                self.state.blue_fallback_role_matrix[:, role_idx]
            )

            # If role is completely unavailable but fallbacks exist, use fallback options
            if not blue_role_available and blue_fallback_available:
                self.state.blue_role_matrix[:, role_idx] = (
                    self.state.blue_fallback_role_matrix[:, role_idx]
                )
                self.state.blue_fallbacks_used += 1
                if DEBUG_PRINT:
                    print(
                        f"Blue team using fallback options for role {self.roles[role_idx]}"
                    )

            # Check red team's role availability
            red_role_available = np.any(self.state.red_role_matrix[:, role_idx])
            red_fallback_available = np.any(
                self.state.red_fallback_role_matrix[:, role_idx]
            )

            # If role is completely unavailable but fallbacks exist, use fallback options
            if not red_role_available and red_fallback_available:
                self.state.red_role_matrix[:, role_idx] = (
                    self.state.red_fallback_role_matrix[:, role_idx]
                )
                self.state.red_fallbacks_used += 1
                if DEBUG_PRINT:
                    print(
                        f"Red team using fallback options for role {self.roles[role_idx]}"
                    )

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
            "blue_role_matrix": self.state.blue_role_matrix.copy(),  # TODO: sometimes hide the ennemy matrix
            "red_role_matrix": self.state.red_role_matrix.copy(),  # TODO: sometimes hide the ennemy matrix
            "blue_roles": blue_roles,
            "red_roles": red_roles,
            "phase": np.array([action_info["phase"]], dtype=np.int8),
            "turn": np.array([action_info["team"]], dtype=np.int8),
            "action_mask": self.get_action_mask(),
            "numerical_patch": np.array([float(self.patch)], dtype=np.float32),
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

        # Get base winrate prediction
        base_winrate = np.float32(
            fetch_blue_side_winrate_prediction(
                np.array(champion_ids, dtype=np.int32), numerical_patch=self.patch
            )
        )

        # Adjust winrate based on fallbacks used (0.15 per fallback)
        FALLBACK_PENALTY = 0.15
        # Blue fallbacks decrease blue's winrate (decrease base_winrate)
        # Red fallbacks increase blue's winrate (increase base_winrate)
        adjusted_winrate = (
            base_winrate
            - (self.state.blue_fallbacks_used * FALLBACK_PENALTY)
            + (self.state.red_fallbacks_used * FALLBACK_PENALTY)
        )

        # Clamp winrate between 0 and 1
        return np.float32(max(0.0, min(1.0, adjusted_winrate)))


def action_mask_fn(env: FlexibleRoleDraftEnv):
    """Action mask function that can be pickled."""
    return env.get_action_mask()
