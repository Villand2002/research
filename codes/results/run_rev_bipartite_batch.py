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


def main() -> None:
    start = time.perf_counter()
    for idx in range(100):
        dataset = _build_dataset(1000 + idx)
        outcome = execute_rev_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                "REV (bipartite) produced infeasible matching on dataset "
                f"{idx}: {violations} violations"
            )
    duration = time.perf_counter() - start

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "rev_bipartite_batch.json"

    payload = {
        "algorithm": "REV (bipartite)",
        "count": 100,
        "duration_seconds": round(duration, 4),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"[REV bipartite] random batch of 100 datasets completed in {duration:.2f}s")


if __name__ == "__main__":
    main()
