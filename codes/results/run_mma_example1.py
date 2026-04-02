import json
import sys
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "codes").is_dir():
            return parent
    return start.parent


ROOT = _find_repo_root(Path(__file__).resolve())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.algorithm.mma import MMASolver


def build_example1():
    agents = ["i1", "i2", "i3"]
    categories = ["c1", "c2"]
    capacities = {"c1": 1, "c2": 1}
    priorities = {
        "c1": ["i2", "i1", "i3"],
        "c2": ["i2", "i3", "i1"],
    }
    return agents, categories, capacities, priorities


def main() -> None:
    agents, categories, capacities, priorities = build_example1()
    solver = MMASolver(agents, categories, capacities, priorities)
    matching = solver.solve()

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "mma_example1.json"

    payload = {"matching": matching}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
