from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.utils import get_action_masks

from utils import DATA_DIR, get_best_device
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
model.learn(total_timesteps=100)

# Save the trained model
model.save(f"{DATA_DIR}/lol_draft_ppo")

# Test the trained agent
obs = env.reset()
for _ in range(1000):
    # Get the action mask
    action_masks = get_action_masks(env)

    # Use the action_masks when predicting the action
    action, _states = model.predict(obs, action_masks=action_masks, deterministic=True)

    # dummy vec env returns a list of observations, rewards, dones, and infos not truncated/terminated
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
