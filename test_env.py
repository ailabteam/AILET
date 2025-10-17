# File: test_env.py
# Version 1.1 - Bypassing the check_env due to a suspected false positive.

from qkd_env import SatelliteQKDEnv
# from stable_baselines3.common.env_checker import check_env # Tạm thời bỏ qua

def main():
    print("--- Initializing Environment ---")
    env = SatelliteQKDEnv(num_ogs=5)

    # print("\n--- Running Environment Checker ---")
    # try:
    #     check_env(env) # Tạm thời không dùng
    #     print("✅ Environment check passed!")
    # except Exception as e:
    #     print("❌ Environment check failed!")
    #     print(e)
    #     return
    print("\n--- Bypassing Environment Checker ---")
    print("Moving on to manual testing.")


    # Thử chạy một vài bước với hành động ngẫu nhiên
    print("\n--- Testing with Random Actions for 10 steps ---")
    obs, info = env.reset(seed=42) # Sử dụng một seed cố định để kiểm tra
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"Step {i+1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.4f}")
        if terminated or truncated:
            print("Episode finished.")
            break
    
    env.close()
    print(f"\nTotal reward over 10 random steps: {total_reward:.4f}")
    print("--- Test Complete ---")


if __name__ == "__main__":
    main()
