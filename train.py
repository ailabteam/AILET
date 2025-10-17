# File: train.py
import os
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from qkd_env import SatelliteQKDEnv

def main():
    # --- Parameters ---
    LOG_DIR = "logs"
    MODEL_DIR = "models"
    TOTAL_TIMESTEPS = 100_000 # Bắt đầu với số bước nhỏ để test
    
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # --- Create Environment ---
    print("Creating the SatelliteQKDEnv...")
    env = SatelliteQKDEnv(num_ogs=5)
    env.reset(seed=42) # Reset lần đầu để khởi tạo

    # --- Setup Model ---
    # Kiểm tra xem GPU có sẵn không
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # PPO là một thuật toán mạnh mẽ, phù hợp cho cả action space rời rạc và liên tục
    model = PPO(
        "MlpPolicy",          # Sử dụng mạng Multi-Layer Perceptron
        env,
        verbose=1,            # In ra thông tin huấn luyện
        tensorboard_log=LOG_DIR,
        device=device
    )

    # --- Setup Callbacks ---
    # Lưu lại model sau mỗi 10,000 bước
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=MODEL_DIR,
        name_prefix="ppo_qkd_model"
    )

    # --- Train the Model ---
    print("--- Starting Training ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=checkpoint_callback
    )
    print("--- Training Finished ---")

    # --- Save the Final Model ---
    final_model_path = os.path.join(MODEL_DIR, "ppo_qkd_model_final.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    env.close()

if __name__ == '__main__':
    main()
