import json
import sys
import time
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "codes").is_dir():
            return parent
    return start.parent


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    precedence = categories[:]
    beneficial_cutoff = max(1, len(categories) // 2)
    beneficial = categories[:beneficial_cutoff]
    return agents, categories, capacities, priorities, precedence, beneficial


def main() -> None:
    start = time.perf_counter()
    for idx in range(100):
        agents, categories, capacities, priorities, precedence, beneficial = _build_scu_inputs(
            2000 + idx
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
        if len(matching) > sum(capacities.values()):
            raise AssertionError("SCU matching exceeded total capacity")
        if len(matching) != len(set(matching.keys())):
            raise AssertionError("SCU assigned an agent more than once")
        for cat in categories:
            assigned = [ag for ag, c in matching.items() if c == cat]
            if len(assigned) > capacities[cat]:
                raise AssertionError(f"Category {cat} exceeded capacity")
            for ag in assigned:
                if ag not in priorities[cat]:
                    raise AssertionError(f"{ag} not eligible for {cat}")
    duration = time.perf_counter() - start

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "scu_batch.json"

    payload = {
        "algorithm": "SCU",
        "count": 100,
        "duration_seconds": round(duration, 4),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"[SCU] random batch of 100 datasets completed in {duration:.2f}s")


if __name__ == "__main__":
    main()
