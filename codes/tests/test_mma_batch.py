import time

import pytest
from codes.algorithm.mma import execute_mma_on_dataset
from codes.data_generation.dataset import Dataset


def _build_dataset(seed: int) -> Dataset:
    return Dataset.build(
        dataset_id=seed,
        num_agents=40,
        num_categories=12,
        capacity_ratio=1.05,
        capacity_std=0.15,
        eligibility_prob=0.6,
        priority_phi=0.85,
        preference_phi=0.75,
        max_preference_length=6,
        seed=seed,
    )


@pytest.mark.slow
def test_mma_random_batch():
    start = time.perf_counter()
    for idx in range(100):
        dataset = _build_dataset(idx)
        outcome = execute_mma_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        assert feasible, f"MMA produced infeasible matching on dataset {idx}: {violations} violations"
    duration = time.perf_counter() - start
    print(f"[MMA] random batch of 100 datasets completed in {duration:.2f}s")
