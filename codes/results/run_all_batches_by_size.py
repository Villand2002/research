import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


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
from codes.batch_shared import (
    BATCH_DATASET_SEEDS,
    BatchDatasetConfig,
    build_batch_dataset,
    build_scu_inputs_from_dataset,
)
from codes.data_generation.dataset import Dataset


def _run_mma(datasets: List[Dataset], progress_every: int = 100) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
        outcome = execute_mma_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                f"MMA produced infeasible matching on dataset {idx}: {violations} violations"
            )
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"[MMA] processed {idx + 1}/{len(datasets)}")
    return time.perf_counter() - start


def _run_rev(datasets: List[Dataset], progress_every: int = 100) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
        outcome = execute_rev_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                f"REV produced infeasible matching on dataset {idx}: {violations} violations"
            )
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"[REV] processed {idx + 1}/{len(datasets)}")
    return time.perf_counter() - start


def _run_rev_bipartite(datasets: List[Dataset], progress_every: int = 100) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
        outcome = execute_rev_bipartite_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        if not feasible:
            raise AssertionError(
                "REV (bipartite) produced infeasible matching on dataset "
                f"{idx}: {violations} violations"
            )
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"[REV (bipartite)] processed {idx + 1}/{len(datasets)}")
    return time.perf_counter() - start


def _run_scu(datasets: List[Dataset], progress_every: int = 100) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
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
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"[SCU] processed {idx + 1}/{len(datasets)}")
    return time.perf_counter() - start


def _run_scu_comb(datasets: List[Dataset], progress_every: int = 100) -> float:
    start = time.perf_counter()
    for idx, dataset in enumerate(datasets):
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
        if progress_every and (idx + 1) % progress_every == 0:
            print(f"[SCUcomb] processed {idx + 1}/{len(datasets)}")
    return time.perf_counter() - start


def _build_datasets(num_agents: int, count: int) -> List[Dataset]:
    config = BatchDatasetConfig(num_agents=num_agents)
    seeds = BATCH_DATASET_SEEDS[:count]
    return [build_batch_dataset(seed, config) for seed in seeds]


def _write_results(
    num_agents: int,
    count: int,
    algorithm_name: str,
    algorithm_key: str,
    duration: float,
) -> None:
    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / f"all_batch_{num_agents}_{algorithm_key}.json"

    payload = {
        "num_agents": num_agents,
        "count": count,
        "algorithm": algorithm_name,
        "duration_seconds": round(duration, 4),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


def _available_algorithms() -> List[Tuple[str, str]]:
    return [
        ("MMA", "mma"),
        ("REV", "rev"),
        ("REV (bipartite)", "rev_bipartite"),
        ("SCU", "scu"),
        ("SCUcomb", "scucomb"),
    ]


def _normalize_algorithms(values: Iterable[str]) -> List[str]:
    if not values:
        return [key for _, key in _available_algorithms()]
    normalized = []
    for value in values:
        normalized.append(value.strip().lower())
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all batch algorithms across sizes.")
    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[50, 100, 1000],
        help="List of agent counts to compare.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=len(BATCH_DATASET_SEEDS),
        help="Number of datasets to generate per size.",
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=[],
        help=(
            "Algorithms to run. Choices: mma, rev, rev_bipartite, scu, scucomb. "
            "Default: all."
        ),
    )
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.count > len(BATCH_DATASET_SEEDS):
        raise ValueError(f"--count must be <= {len(BATCH_DATASET_SEEDS)}")

    algo_keys = _normalize_algorithms(args.algorithms)
    available = {key for _, key in _available_algorithms()}
    unknown = sorted(set(algo_keys) - available)
    if unknown:
        raise ValueError(f"Unknown algorithms: {', '.join(unknown)}")

    for num_agents in args.sizes:
        datasets = _build_datasets(num_agents, args.count)
        for name, key in _available_algorithms():
            if key not in algo_keys:
                continue
            if key == "mma":
                duration = _run_mma(datasets)
            elif key == "rev":
                duration = _run_rev(datasets)
            elif key == "rev_bipartite":
                duration = _run_rev_bipartite(datasets)
            elif key == "scu":
                duration = _run_scu(datasets)
            elif key == "scucomb":
                duration = _run_scu_comb(datasets)
            else:
                raise AssertionError(f"Unhandled algorithm key: {key}")
            _write_results(num_agents, args.count, name, key, duration)
            print(
                f"[{name}] size={num_agents} batch of {len(datasets)} datasets "
                f"completed in {duration:.2f}s"
            )


if __name__ == "__main__":
    main()
