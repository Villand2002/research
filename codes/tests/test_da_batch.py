import time

import pytest

from codes.algorithm.da import execute_da_on_dataset
from codes.batch_shared import BATCH_DATASET_SEEDS, build_batch_dataset


@pytest.mark.slow
def test_da_random_batch():
    count = len(BATCH_DATASET_SEEDS)
    start = time.perf_counter()
    for seed in BATCH_DATASET_SEEDS:
        dataset = build_batch_dataset(seed)
        outcome = execute_da_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        assert feasible, f"DA produced infeasible matching on dataset {seed}: {violations} violations"
    duration = time.perf_counter() - start
    print(f"[DA] random batch of {count} datasets completed in {duration:.2f}s")
