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

from codes.agent import Agent, Agents, Category
from codes.algorithm.rev import rev_algorithm


def build_example1():
    a1 = Agent(agent_id=1, acceptable_categories=[1])
    a2 = Agent(agent_id=2, acceptable_categories=[1, 2])
    a3 = Agent(agent_id=3, acceptable_categories=[2])
    agents_obj = Agents(agents=[a1, a2, a3], agent_number=3)

    c1 = Category(category_id=1, capacity=1, priority=[2, 1])
    c2 = Category(category_id=2, capacity=1, priority=[2, 3])

    c1.eligible_agents = [2, 1]
    c2.eligible_agents = [2, 3]

    categories_obj = [c1, c2]
    return agents_obj, categories_obj


def main() -> None:
    agents_obj, categories_obj = build_example1()
    matching, banned = rev_algorithm(agents_obj, categories_obj)
    matching_dict = {agent_id: category_id for agent_id, category_id in matching}

    result_dir = ROOT / "result _csv"
    result_dir.mkdir(parents=True, exist_ok=True)
    output_path = result_dir / "rev_example1.json"

    payload = {
        "matching": matching_dict,
        "banned": sorted(banned),
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
