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


def main() -> None:
    start = time.perf_counter()
    for idx in range(100):
        dataset = _build_dataset(idx)
        outcome = execute_mma_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                f"MMA produced infeasible matching on dataset {idx}: {violations} violations"
            )
    duration = time.perf_counter() - start

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "mma_batch.json"

    payload = {
        "algorithm": "MMA",
        "count": 100,
        "duration_seconds": round(duration, 4),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"[MMA] random batch of 100 datasets completed in {duration:.2f}s")


if __name__ == "__main__":
    main()
