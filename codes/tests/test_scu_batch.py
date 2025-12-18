import time

import pytest
from codes.algorithm.scu import SCUSolver
from codes.data_generation.dataset import Dataset


def _build_scu_inputs(seed: int):
    dataset = Dataset.build(
        dataset_id=seed,
        num_agents=30,
        num_categories=8,
        capacity_ratio=1.0,
        capacity_std=0.1,
        eligibility_prob=0.6,
        priority_phi=0.85,
        preference_phi=0.75,
        max_preference_length=6,
        seed=seed,
    )
    agents = [f"i{ag.agent_id}" for ag in dataset.agents]
    categories = [f"c{cat.category_id}" for cat in dataset.categories]
    capacities = {f"c{cat.category_id}": cat.capacity for cat in dataset.categories}
    priorities = {
        f"c{cat.category_id}": [f"i{aid}" for aid in cat.priority] for cat in dataset.categories
    }
    precedence = categories[:]  # simple precedence by id
    beneficial_cutoff = max(1, len(categories) // 2)
    beneficial = categories[:beneficial_cutoff]
    return agents, categories, capacities, priorities, precedence, beneficial


@pytest.mark.slow
def test_scu_random_batch():
    start = time.perf_counter()
    for idx in range(100):
        agents, categories, capacities, priorities, precedence, beneficial = _build_scu_inputs(2000 + idx)
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
    print(f"[SCU] random batch of 100 datasets completed in {duration:.2f}s")
