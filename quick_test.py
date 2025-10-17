# File: quick_test.py
# A quick pre-flight check for both environment scenarios before long training runs.

from qkd_env import SatelliteQKDEnv
import numpy as np

def test_scenario(dynamic_weather: bool):
    """Tests a single environment scenario for basic functionality."""
    scenario_name = "Dynamic" if dynamic_weather else "Static"
    print(f"\n--- Testing {scenario_name} Environment ---")

    try:
        # 1. Initialization
        env = SatelliteQKDEnv(num_ogs=5, dynamic_weather=dynamic_weather)
        print("✅ Environment initialized successfully.")

        # 2. Reset
        obs, info = env.reset(seed=42)
        print("✅ Environment reset successfully.")
        print(f"  Observation shape: {obs.shape}")
        
        # Check if observation shape matches the defined space
        if obs.shape != env.observation_space.shape:
             print(f"❌ MISMATCH! Obs shape is {obs.shape} but space is {env.observation_space.shape}")
             return False
        else:
            print(f"  Observation shape matches space.")

        # 3. Step
        # Run a few steps with random actions
        print("Running 5 random steps...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # For the dynamic env, let's check if the weather is changing
            if dynamic_weather:
                env.render() # The render function will print cloud status
        
        print("✅ 5 steps executed without errors.")
        env.close()
        return True

    except Exception as e:
        print(f"❌ An error occurred during the {scenario_name} test!")
        print(e)
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*50)
    print("Starting Pre-flight Check for SatelliteQKDEnv")
    print("="*50)

    static_ok = test_scenario(dynamic_weather=False)
    dynamic_ok = test_scenario(dynamic_weather=True)

    print("\n--- Test Summary ---")
    if static_ok and dynamic_ok:
        print("✅ All tests passed. You are cleared for training!")
    else:
        print("❌ One or more tests failed. Please review the errors above before training.")

if __name__ == "__main__":
    main()
