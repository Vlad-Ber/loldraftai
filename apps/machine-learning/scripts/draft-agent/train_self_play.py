# draft_agent/train_self_play.py

import os
import warnings
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
import wandb
from wandb.integration.sb3 import WandbCallback
from pathlib import Path
import pickle
from datetime import datetime
from typing import List

from utils import DATA_DIR
from utils.match_prediction import get_best_device
from utils.rl import ROLE_CHAMPIONS_PATH
from utils.rl.env import FixedRoleDraftEnv, FlexibleRoleDraftEnv
from utils.rl.self_play import ModelPool, SelfPlayWithPoolWrapper
from utils.rl.env import action_mask_fn
from utils.match_prediction import PREPARED_DATA_DIR

# sb3_contrib is not updated to latest api, this is the message we are ignoring:
# WARN: env.get_action_mask to get variables from other wrappers is deprecated and will be removed in v1.0
warnings.filterwarnings("ignore", message=".*env.get_action_mask.*")


def get_latest_patches(n_patches: int = 5) -> List[int]:
    """
    Load patch mapping and return the n latest numerical patches.

    Args:
        n_patches: Number of latest patches to return

    Returns:
        List of numerical patch values, sorted from newest to oldest
    """
    patch_mapping_path = Path(PREPARED_DATA_DIR) / "patch_mapping.pkl"
    with open(patch_mapping_path, "rb") as f:
        patch_data = pickle.load(f)

    # Get unique raw patch numbers
    raw_patches = sorted(set(patch_data["mapping"].keys()))

    # Return the n latest patches (highest numbers)
    return raw_patches[-n_patches:]


def train_self_play(
    num_iterations: int = 10,
    timesteps_per_iteration: int = 50_000,
    num_envs: int = 32,
    pool_size: int = 5,
    save_dir: str = f"{DATA_DIR}/self_play_models",
    random_opponent_prob: float = 0.1,
    latest_model_prob: float = 0.5,
    use_wandb: bool = True,
):
    run = None
    if use_wandb:
        # Initialize W&B run
        run = wandb.init(
            project="draft-agent",
            config={
                "num_iterations": num_iterations,
                "timesteps_per_iteration": timesteps_per_iteration,
                "num_envs": num_envs,
                "pool_size": pool_size,
                "random_opponent_prob": random_opponent_prob,
                "latest_model_prob": latest_model_prob,
            },
            sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
            save_code=True,  # optional
        )
        # Set up a single tensorboard directory for all iterations
        tensorboard_log = f"runs/{run.id}"
    else:
        tensorboard_log = None

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model pool
    model_pool = ModelPool(
        save_dir=save_dir,
        pool_size=pool_size,
        latest_model_prob=latest_model_prob,
        random_opponent_prob=random_opponent_prob,
    )

    patches = get_latest_patches(5)
    print(f"Using latest patches: {patches}")

    role_rates_path = os.path.join(
        os.path.dirname(ROLE_CHAMPIONS_PATH), "champion_role_rates.json"
    )

    def make_env(rank, model_pool, agent_side="random"):
        def _init():
            # env = FixedRoleDraftEnv(patches=patches)
            # TODO: add patches to FlexibleRoleDraftEnv
            env = FlexibleRoleDraftEnv()
            env = SelfPlayWithPoolWrapper(env, model_pool, agent_side)
            env = ActionMasker(env, action_mask_fn)
            return env

        return _init

    agent_side: str = "random"  # We always want to play from both sides in training
    # Create vectorized environment
    env = SubprocVecEnv([make_env(i, model_pool, agent_side) for i in range(num_envs)])

    device = get_best_device()

    # Initialize the agent
    model = MaskablePPO(
        "MultiInputPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=tensorboard_log,
    )

    # Training loop
    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")

        try:
            # Train the agent with optional W&B callback
            callback = (
                WandbCallback(
                    gradient_save_freq=100,
                    model_save_path=f"{save_dir}/models/{run.id}",
                    verbose=2,
                )
                if use_wandb
                else None
            )

            model.learn(
                total_timesteps=timesteps_per_iteration,
                progress_bar=True,
                callback=callback,
                reset_num_timesteps=False,  # Important: Don't reset timesteps between iterations
                tb_log_name="PPO",  # Use same name for all iterations
            )

            # Update the pool with the latest model
            model_pool.update_pool(model, iteration)

        except Exception as e:
            print(f"Training interrupted at iteration {iteration + 1}: {e}")
            break

    # Save the final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    # Save timestamped to final_models/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_models_dir = os.path.join(save_dir, "final_models")
    os.makedirs(final_models_dir, exist_ok=True)
    final_model_path = os.path.join(final_models_dir, f"{timestamp}.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path} and {final_path}")

    # Finish W&B run if it was initialized
    if run is not None:
        run.finish()

    return model


if __name__ == "__main__":
    trained_model = train_self_play(
        num_iterations=15,
        # timesteps_per_iteration=2000,
        timesteps_per_iteration=500_000,
        num_envs=32,
        pool_size=5,
        random_opponent_prob=0.05,
    )
