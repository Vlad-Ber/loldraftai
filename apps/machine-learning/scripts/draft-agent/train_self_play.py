# train_self_play.py

import os
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from sb3_contrib.common.wrappers import ActionMasker
from utils import DATA_DIR
from utils.match_prediction import get_best_device
from utils.rl.env import FixedRoleDraftEnv
from utils.rl.self_play import ModelPool, SelfPlayWithPoolWrapper, action_mask_fn


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
    num_envs: int = 8,
    pool_size: int = 5,
    save_dir: str = f"{DATA_DIR}/self_play_models",
    agent_side: str = "random",
):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Initialize model pool
    model_pool = ModelPool(
        save_dir=save_dir,
        pool_size=pool_size,
        latest_model_prob=0.5,
        random_opponent_prob=0.1,
    )

    # Create vectorized environment
    env = SubprocVecEnv([make_env(i, model_pool, agent_side) for i in range(num_envs)])

    device = get_best_device()

    # Initialize the agent
    model = MaskablePPO("MultiInputPolicy", env, verbose=1, device=device)

    # Training loop
    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")

        try:
            # Train the agent
            model.learn(total_timesteps=timesteps_per_iteration, progress_bar=True)

            # Update the pool with the latest model
            model_pool.update_pool(model, iteration)

        except Exception as e:
            print(f"Training interrupted at iteration {iteration + 1}: {e}")
            break

    # Save the final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"Final model saved to {final_path}")

    return model


if __name__ == "__main__":
    trained_model = train_self_play(
        num_iterations=10,
        timesteps_per_iteration=50_000,
        num_envs=8,
        pool_size=5,
        agent_side="random",  # or "blue" or "red"
    )
