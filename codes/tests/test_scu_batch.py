import time

import pytest

from codes.algorithm.scu import SCUSolver
from codes.batch_shared import BATCH_DATASET_SEEDS, build_batch_dataset, build_scu_inputs_from_dataset


@pytest.mark.slow
def test_scu_random_batch():
    count = len(BATCH_DATASET_SEEDS)
    start = time.perf_counter()
    for seed in BATCH_DATASET_SEEDS:
        dataset = build_batch_dataset(seed)
        agents, categories, capacities, priorities, precedence, beneficial = (
            build_scu_inputs_from_dataset(dataset)
        )
        solver = SCUSolver(
            agents,
            categories,
            capacities,
            priorities,
            precedence,
            beneficial,
        )
        matching = solver.solve()
        assert len(matching) <= sum(capacities.values())
        # No agent assigned twice and all assignments respect capacity/priority membership
        assert len(matching) == len(set(matching.keys()))
        for cat in categories:
            assigned = [ag for ag, c in matching.items() if c == cat]
            assert len(assigned) <= capacities[cat], f"Category {cat} exceeded capacity"
            for ag in assigned:
                assert ag in priorities[cat], f"{ag} not eligible for {cat}"
    duration = time.perf_counter() - start
    print(f"[SCU] random batch of {count} datasets completed in {duration:.2f}s")
