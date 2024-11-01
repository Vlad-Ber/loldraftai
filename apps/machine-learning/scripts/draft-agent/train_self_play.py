# draft_agent/train_self_play.py

import os
import warnings
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from utils import DATA_DIR
from utils.match_prediction import get_best_device
from utils.rl.env import FixedRoleDraftEnv
from utils.rl.self_play import ModelPool, SelfPlayWithPoolWrapper
from utils.rl.env import action_mask_fn
import wandb
from wandb.integration.sb3 import WandbCallback

# sb3_contrib is not updated to latest api, this is the message we are ignoring:
# WARN: env.get_action_mask to get variables from other wrappers is deprecated and will be removed in v1.0
warnings.filterwarnings("ignore", message=".*env.get_action_mask.*")


def make_env(rank, model_pool, agent_side="random"):
    def _init():
        env = FixedRoleDraftEnv()
        env = SelfPlayWithPoolWrapper(env, model_pool, agent_side)
        env = ActionMasker(env, action_mask_fn)
        return env

    return _init


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

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model pool
    model_pool = ModelPool(
        save_dir=save_dir,
        pool_size=pool_size,
        latest_model_prob=latest_model_prob,
        random_opponent_prob=random_opponent_prob,
    )

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
        tensorboard_log=f"runs/{run.id}" if use_wandb else None,
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
            )

            # Update the pool with the latest model
            model_pool.update_pool(model, iteration)

        except Exception as e:
            print(f"Training interrupted at iteration {iteration + 1}: {e}")
            break

    # Save the final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    # Finish W&B run if it was initialized
    if run is not None:
        run.finish()

    return model


if __name__ == "__main__":
    trained_model = train_self_play(
        num_iterations=15,
        timesteps_per_iteration=250_000,
        num_envs=32,
        pool_size=10,
        random_opponent_prob=0.05,
    )
