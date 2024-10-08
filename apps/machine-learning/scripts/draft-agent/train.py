import pickle
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from utils import MODEL_CONFIG_PATH, get_best_device, DATA_DIR
from utils.rl.env import create_solo_queue_draft_order
from utils.rl import fetch_blue_side_winrate_prediction

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
            }
        )

        # Define the draft order for solo queue
        self.draft_order = create_solo_queue_draft_order()
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

    def step(self, action):
        if self.done:
            raise Exception("Cannot call step() on a done environment")

        # Get current action info
        action_info = self.draft_order[self.current_step]
        current_team = action_info["team"]
        phase = action_info["phase"]
        current_role_index = action_info.get("role_index", None)

        # Process the action (pick or ban)
        valid = self._process_action(
            action, phase, current_team, current_role_index
        )
        if not valid:
            # Invalid action, penalize
            reward = -1  # Penalty for invalid action
            terminated = True
            truncated = False
            info = {"reason": "invalid_action"}
            observation = self._get_obs()
            self.done = True
            return observation, reward, terminated, truncated, info

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
            return False  # Invalid action

        if phase == 0:
            self._ban_champion(action)
        elif phase == 1:
            self._pick_champion(action, current_team)
        elif phase == 2:
            return self._assign_role(action, current_team, current_role_index)
        else:
            raise ValueError(f"Unknown phase: {phase}")

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
            return False  # Invalid action (champion not in picks or already assigned)
        ordered_picks[current_role_index][action] = 1

    def _get_team_picks(self, current_team):
        return (self.blue_picks, self.blue_ordered_picks) if current_team == 0 else (self.red_picks, self.red_ordered_picks)

    def _update_state(self):
        self.current_step += 1
        if self.current_step >= len(self.draft_order):
            self.done = True
    
    def _get_action_info(self):
        return self.draft_order[self.current_step]

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

        phase = self.draft_order[self.current_step]["phase"]
        return {
            "available_champions": self.available_champions.copy(),
            "blue_picks": self.blue_picks.copy(),
            "red_picks": self.red_picks.copy(),
            "blue_ordered_picks": blue_ordered_picks,
            "red_ordered_picks": red_ordered_picks,
            "phase": np.array([phase], dtype=np.int8),
            "turn": np.array([current_turn], dtype=np.int8),
            "current_role": np.array([current_role_index], dtype=np.int8),
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
        # TODO: reward should depend on agent's team

        return reward

    def render(self):
        pass  # No rendering needed

    def close(self):
        pass  # Nothing to close


# Wrapper for self-play
class SelfPlayWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, *, seed=None, options=None):
        # Pass seed and options to the underlying environment
        observation, info = self.env.reset(seed=seed, options=options)
        return observation, info

    def step(self, action):
        # Get current action info
        action_info = self.env._get_action_info()

        # Check if it's the agent's turn (assume agent is blue team)
        if action_info["team"] == 0:
            # Agent's turn
            observation, reward, terminated, truncated, info = self.env.step(action)
        else:
            # Opponent's turn, use random valid action
            valid_actions = self._get_valid_actions()
            opponent_action = self.np_random.choice(valid_actions)
            observation, _, terminated, truncated, info = self.env.step(opponent_action)
            reward = 0  # No reward for opponent's action

        # After opponent's turn, if it's not done, check if it's agent's turn again
        while (
            not terminated
            and not truncated
            and self.env.current_step < len(self.env.draft_order)
            and self.env.draft_order[self.env.current_step]["team"] != 0
        ):
            # It's opponent's turn again
            valid_actions = self._get_valid_actions()
            opponent_action = self.np_random.choice(valid_actions)
            observation, _, terminated, truncated, info = self.env.step(opponent_action)

        if terminated or truncated:
            # Get final reward
            if self.env.current_step >= len(self.env.draft_order):
                reward = self.env._calculate_reward()
            else:
                reward = 0 # terminated because of invalid action

        return observation, reward, terminated, truncated, info

    def _get_valid_actions(self):
        action_info = self.env._get_action_info()
        team = action_info["team"]
        phase = action_info["phase"]

        if phase == 0:
            # Valid actions are available champions
            valid_actions = np.where(self.env.available_champions == 1)[0]
        elif phase == 1:
            # Valid actions are available champions
            valid_actions = np.where(self.env.available_champions == 1)[0]
        elif phase == 2:
            # Valid actions are the champions picked by the team but not yet assigned to a role
            if team == 0:
                picks = self.env.blue_picks
                ordered_picks = self.env.blue_ordered_picks
            else:
                picks = self.env.red_picks
                ordered_picks = self.env.red_ordered_picks
            unassigned_champions = picks - ordered_picks
            valid_actions = np.where(np.sum(unassigned_champions, axis=0) == 1)[0]
        else:
            # Should not happen
            raise ValueError(f"State with no valid actions: {self.env.current_step}")
        return valid_actions


# Create and wrap the environment
env = LoLDraftEnv()
env = SelfPlayWrapper(env)
env = DummyVecEnv([lambda: env])

device = get_best_device()

# Initialize the agent
model = PPO("MultiInputPolicy", env, verbose=1, device=device)

# Train the agent
model.learn(total_timesteps=1000)

# Save the trained model
model.save(f"{DATA_DIR}/lol_draft_ppo")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    
    # dummy vec env returns a list of observations, rewards, dones, and infos not truncated/terminated
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
