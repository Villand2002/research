import time

import pytest

from codes.algorithm.rev_bipartite import execute_rev_on_dataset
from codes.data_generation.dataset import Dataset


def _build_dataset(seed: int) -> Dataset:
    return Dataset.build(
        dataset_id=seed,
        num_agents=35,
        num_categories=10,
        capacity_ratio=1.0,
        capacity_std=0.2,
        eligibility_prob=0.55,
        priority_phi=0.8,
        preference_phi=0.7,
        max_preference_length=5,
        seed=seed,
    )


@pytest.mark.slow
def test_rev_bipartite_random_batch():
    start = time.perf_counter()
    for idx in range(100):
        dataset = _build_dataset(1000 + idx)
        outcome = execute_rev_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        assert feasible, (
            "REV (bipartite) produced infeasible matching on dataset "
            f"{idx}: {violations} violations"
        )
    duration = time.perf_counter() - start
    print(f"[REV bipartite] random batch of 100 datasets completed in {duration:.2f}s")
