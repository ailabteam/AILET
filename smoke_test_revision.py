# File: smoke_test_revision.py
# Lightweight smoke test for the revision code. Exercises the parameterized
# environment and the heuristic/hybrid policies WITHOUT torch/SB3, so it runs in a
# minimal numpy+gymnasium+skyfield environment. The DRL path is validated
# separately on the GPU server.

import numpy as np
from qkd_env import SatelliteQKDEnv
from policies import GreedyPolicy, HybridPolicy, RandomPolicy
from evaluate_revision import evaluate_policy


def test_env_shapes_and_costs():
    for scenario in ["static", "dynamic", "realistic"]:
        env = SatelliteQKDEnv(num_ogs=5, scenario=scenario)
        obs, _ = env.reset(seed=0)
        assert obs.shape == env.observation_space.shape, scenario
        env.close()
    # cost=0 must not raise (division-by-zero guard) and must run a full episode
    env = SatelliteQKDEnv(num_ogs=5, scenario="realistic", switching_cost_minutes=0)
    obs, _ = env.reset(seed=0)
    term = trunc = False
    steps = 0
    while not (term or trunc):
        obs, r, term, trunc, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == 1440, steps  # 24h at 1-minute steps
    env.close()
    print("[OK] env shapes + cost=0 guard + episode length")


def test_hybrid_degenerates_to_greedy():
    """Hybrid with margin=0 must reproduce Greedy exactly (same actions)."""
    env = SatelliteQKDEnv(num_ogs=5, scenario="realistic", switching_cost_minutes=2)
    g, h = GreedyPolicy(), HybridPolicy(margin_deg=0.0)
    obs, _ = env.reset(seed=7); g.reset(); h.reset()
    term = trunc = False
    while not (term or trunc):
        ag, ah = g.act(obs, env), h.act(obs, env)
        assert ag == ah, (ag, ah)
        obs, _, term, trunc, _ = env.step(ag)
    env.close()
    print("[OK] Hybrid(margin=0) == Greedy")


def test_hybrid_close_to_greedy_and_beats_random():
    """Under a real switching cost the cost-aware Hybrid tracks Greedy closely
    (the empirical finding is that hysteresis gives no material gain here because
    switches are geometry-forced), and both clearly beat Random."""
    seeds = list(range(1000, 1005))
    g = evaluate_policy("realistic", GreedyPolicy(), seeds, switching_cost_minutes=2)
    h = evaluate_policy("realistic", HybridPolicy(margin_deg=15.0), seeds,
                        switching_cost_minutes=2)
    rnd = evaluate_policy("realistic", RandomPolicy(), seeds, switching_cost_minutes=2)
    print(f"  Greedy mean={g.mean():,.1f}  Hybrid mean={h.mean():,.1f}  "
          f"Random mean={rnd.mean():,.1f}")
    assert abs(h.mean() - g.mean()) < 0.10 * g.mean(), "Hybrid should track Greedy"
    assert g.mean() > rnd.mean(), "Greedy should beat Random"
    print("[OK] Hybrid ~ Greedy, both > Random under switching cost")


if __name__ == "__main__":
    test_env_shapes_and_costs()
    test_hybrid_degenerates_to_greedy()
    test_hybrid_close_to_greedy_and_beats_random()
    print("\nAll smoke tests passed.")
