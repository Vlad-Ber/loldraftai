from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker

from utils import DATA_DIR
from utils.match_prediction import get_best_device
from utils.rl.visualizer import integrate_with_env
from utils.rl.env import LoLDraftEnv, SelfPlayWrapper, action_mask_fn

# Create and wrap the environment
env = LoLDraftEnv()
env = integrate_with_env(LoLDraftEnv)()
env = SelfPlayWrapper(env)
env = ActionMasker(env, action_mask_fn)
env = DummyVecEnv([lambda: env])

device = get_best_device()

# Initialize the agent
model = MaskablePPO("MultiInputPolicy", env, verbose=1, device=device)

# Train the agent
model.learn(total_timesteps=10000, progress_bar=True)

# Save the trained model
model.save(f"{DATA_DIR}/lol_draft_ppo")
