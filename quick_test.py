# File: quick_test.py
# Version 2.0 - A quick pre-flight check for all three environment scenarios.
# Verifies initialization, reset, and step functionality before long training runs.

from qkd_env import SatelliteQKDEnv
import numpy as np

def test_scenario(scenario: str):
    """
    Tests a single environment scenario for basic functionality.
    
    :param scenario: The scenario to test ("static", "dynamic", or "realistic").
    """
    print(f"\n--- Testing '{scenario.upper()}' Environment ---")

    try:
        # 1. Initialization
        env = SatelliteQKDEnv(num_ogs=5, scenario=scenario)
        print(f"✅ Environment initialized successfully.")

        # 2. Reset
        obs, info = env.reset(seed=42)
        print(f"✅ Environment reset successfully.")
        
        # 3. Space and Observation Shape Check
        print(f"  Observation shape: {obs.shape}")
        print(f"  Expected space shape: {env.observation_space.shape}")
        
        if obs.shape != env.observation_space.shape:
             print(f"❌ MISMATCH! Observation shape {obs.shape} does not match space shape {env.observation_space.shape}")
             return False
        else:
            print(f"✅ Observation shape matches the defined space.")

        # 4. Step Execution
        print("Running 5 random steps to check for runtime errors...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Use the render function for a quick status check
            print(f"  Step {i+1}: Action={action}, Reward={reward:.2f}")
            env.render()
        
        print(f"✅ 5 steps executed without errors.")
        env.close()
        return True

    except Exception as e:
        print(f"❌ An error occurred during the '{scenario.upper()}' test!")
        # Print the full error traceback for detailed debugging
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    Main function to run the pre-flight check on all scenarios.
    """
    print("="*60)
    print("      Starting Pre-flight Check for SatelliteQKDEnv v3.0      ")
    print("="*60)

    # List of all scenarios to be tested
    scenarios = ["static", "dynamic", "realistic"]
    results = {}

    for scn in scenarios:
        results[scn] = test_scenario(scn)

    print("\n" + "="*60)
    print("--- Test Summary ---")
    
    all_passed = True
    for scn, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"Scenario '{scn.upper()}': {status}")
        if not passed:
            all_passed = False

    print("="*60)
    if all_passed:
        print("✅ All tests passed. You are cleared for the full training pipeline!")
    else:
        print("❌ One or more tests failed. Please review the errors above before training.")

if __name__ == "__main__":
    main()
