import time

import pytest

from codes.algorithm.rev_bipartite import execute_rev_on_dataset
from codes.batch_shared import BATCH_DATASET_SEEDS, build_batch_dataset


@pytest.mark.slow
def test_rev_bipartite_random_batch():
    count = len(BATCH_DATASET_SEEDS)
    start = time.perf_counter()
    for seed in BATCH_DATASET_SEEDS:
        dataset = build_batch_dataset(seed)
        outcome = execute_rev_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        assert feasible, (
            "REV (bipartite) produced infeasible matching on dataset "
            f"{seed}: {violations} violations"
        )
    duration = time.perf_counter() - start
    print(f"[REV bipartite] random batch of {count} datasets completed in {duration:.2f}s")
