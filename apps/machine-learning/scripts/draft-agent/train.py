# draft_agent/train.py

from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker

from utils import DATA_DIR
from utils.match_prediction import get_best_device
from utils.rl.env import LoLDraftEnv, SelfPlayWrapper, action_mask_fn, FixedRoleDraftEnv

# Number of parallel environments
NUM_ENVS = 32 


def make_env(rank):
    def _init():
        env = FixedRoleDraftEnv()
        env = SelfPlayWrapper(env)
        env = ActionMasker(env, action_mask_fn)
        return env

    return _init


def main():
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i) for i in range(NUM_ENVS)])

    device = get_best_device()

    # Initialize the agent
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, device=device)

    try:
        # Train the agent
        model.learn(total_timesteps=50_000, progress_bar=True)
    except Exception as e:
        print(f"Training interrupted: {e}")
    finally:
        # Save the trained model (will run even if there's an error)
        print("Saving model checkpoint...")
        model.save(f"{DATA_DIR}/lol_draft_ppo")
        print("Model saved successfully")


if __name__ == "__main__":
    main()
