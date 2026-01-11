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

from codes.algorithm.scu import SCUSolver


def build_example4():
    agents = ["i1", "i2", "i3", "i4", "i5", "i6"]
    categories = ["c1", "c2", "c3"]
    capacities = {"c1": 1, "c2": 1, "c3": 1}
    priorities = {
        "c1": ["i2", "i1", "i3"],
        "c2": ["i2", "i4", "i6", "i3", "i5", "i1"],
        "c3": ["i5", "i4", "i6"],
    }
    precedence_order = ["c1", "c2", "c3"]
    beneficial_categories = ["c1", "c3"]
    return (
        agents,
        categories,
        capacities,
        priorities,
        precedence_order,
        beneficial_categories,
    )


def main() -> None:
    (
        agents,
        categories,
        capacities,
        priorities,
        precedence_order,
        beneficial_categories,
    ) = build_example4()
    solver = SCUSolver(
        agents,
        categories,
        capacities,
        priorities,
        precedence_order,
        beneficial_categories,
    )
    matching = solver.solve()

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "scu_example4.json"

    payload = {"matching": matching}
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
