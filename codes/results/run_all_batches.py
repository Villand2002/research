import json
import sys
import time
from pathlib import Path
from typing import Dict, List


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "codes").is_dir():
            return parent
    return start.parent


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.algorithm.mma import execute_mma_on_dataset
from codes.algorithm.rev import execute_rev_on_dataset
from codes.algorithm.rev_bipartite import execute_rev_on_dataset as execute_rev_bipartite_on_dataset
from codes.algorithm.scu import SCUSolver
from codes.algorithm.scu_comb import SCUcomb
from codes.batch_shared import BATCH_DATASET_SEEDS, build_batch_dataset, build_scu_inputs_from_dataset
from codes.data_generation.dataset import Dataset


def _run_mma(datasets: List[Dataset]) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
        outcome = execute_mma_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                f"MMA produced infeasible matching on dataset {idx}: {violations} violations"
            )
    return time.perf_counter() - start


def _run_rev(datasets: List[Dataset]) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
        outcome = execute_rev_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                f"REV produced infeasible matching on dataset {idx}: {violations} violations"
            )
    return time.perf_counter() - start


def _run_rev_bipartite(datasets: List[Dataset]) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
        outcome = execute_rev_bipartite_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                "REV (bipartite) produced infeasible matching on dataset "
                f"{idx}: {violations} violations"
            )
    return time.perf_counter() - start


def _run_scu(datasets: List[Dataset]) -> float:
    start = time.perf_counter()
    for dataset in datasets:
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
    return time.perf_counter() - start


def _run_scu_comb(datasets: List[Dataset]) -> float:
    start = time.perf_counter()
    for dataset in datasets:
        agents, categories, capacities, priorities, precedence, beneficial = (
            build_scu_inputs_from_dataset(dataset)
        )
        solver = SCUcomb(
            agents,
            categories,
            capacities,
            priorities,
            precedence,
            beneficial,
        )
        matching = solver.solve()
        if len(matching) > sum(capacities.values()):
            raise AssertionError("SCUcomb matching exceeded total capacity")
        if len(matching) != len(set(matching.keys())):
            raise AssertionError("SCUcomb assigned an agent more than once")
        for cat in categories:
            assigned = [ag for ag, c in matching.items() if c == cat]
            if len(assigned) > capacities[cat]:
                raise AssertionError(f"Category {cat} exceeded capacity")
            for ag in assigned:
                if ag not in priorities[cat]:
                    raise AssertionError(f"{ag} not eligible for {cat}")
    return time.perf_counter() - start


def main() -> None:
    datasets = [build_batch_dataset(seed) for seed in BATCH_DATASET_SEEDS]

    durations: Dict[str, float] = {
        "MMA": _run_mma(datasets),
        "REV": _run_rev(datasets),
        "REV (bipartite)": _run_rev_bipartite(datasets),
        "SCU": _run_scu(datasets),
        "SCUcomb": _run_scu_comb(datasets),
    }

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "all_batch.json"

    payload = {
        "count": len(datasets),
        "duration_seconds": {name: round(value, 4) for name, value in durations.items()},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    for name, value in durations.items():
        print(f"[{name}] random batch of {len(datasets)} datasets completed in {value:.2f}s")


if __name__ == "__main__":
    main()
