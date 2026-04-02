from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from codes.agent import Agents, Category, Outcome
from codes.data_generation.dataset import Dataset


def build_reserve_induced_preferences(
    agents: List[Any],
    categories_obj: List[Category],
    precedence: List[Any],
) -> Dict[Any, List[Any]]:
    """
    Build reserve-induced preference lists from precedence.
    Each agent proposes only to categories that are both acceptable and eligible,
    ordered by precedence.
    """
    category_by_id = {cat.category_id: cat for cat in categories_obj}
    preferences: Dict[Any, List[Any]] = {}

    for ag in agents:
        pref: List[Any] = []
        acceptable = set(ag.acceptable_categories)
        for cat_id in precedence:
            cat = category_by_id.get(cat_id)
            if cat is None:
                continue
            if cat_id in acceptable and ag.agent_id in cat.eligible_agents:
                pref.append(cat_id)
        preferences[ag.agent_id] = pref

    return preferences


def da_algorithm(
    agents_obj: Agents,
    categories_obj: List[Category],
    precedence: Optional[List[Any]] = None,
) -> List[Tuple[Any, Any]]:
    """
    Pathak et al. (2023) の DA（SCUと同値）

    precedence order を agent preference に埋め込んだ
    agent-proposing DA
    """

    agents = agents_obj.agents
    category_by_id = {cat.category_id: cat for cat in categories_obj}

    if precedence is None:
        precedence = [cat.category_id for cat in categories_obj]

    preferences = build_reserve_induced_preferences(
        agents,
        categories_obj,
        precedence,
    )

    category_assignments: Dict[Any, List[Any]] = {
        cat.category_id: [] for cat in categories_obj
    }

    priority_rank: Dict[Any, Dict[Any, int]] = {
        cat.category_id: {
            agent_id: idx for idx, agent_id in enumerate(cat.priority)
        }
        for cat in categories_obj
    }

    next_choice_idx: Dict[Any, int] = {ag.agent_id: 0 for ag in agents}
    matching: Dict[Any, Any] = {}
    free_agents = deque(ag.agent_id for ag in agents)

    while free_agents:
        agent_id = free_agents.popleft()
        pref_list = preferences[agent_id]

        matched = False

        while next_choice_idx[agent_id] < len(pref_list):

            cat_id = pref_list[next_choice_idx[agent_id]]
            next_choice_idx[agent_id] += 1
            category = category_by_id[cat_id]

            current = category_assignments[cat_id]
            candidates = current + [agent_id]
            ranks = priority_rank.get(cat_id, {})

            # priority順にソート
            candidates.sort(key=lambda aid: ranks.get(aid, float("inf")))

            accepted = candidates[: category.capacity]
            rejected = [aid for aid in candidates if aid not in accepted]

            category_assignments[cat_id] = accepted

            for aid in accepted:
                matching[aid] = cat_id

            for aid in rejected:
                if matching.get(aid) == cat_id:
                    del matching[aid]

                if aid != agent_id and next_choice_idx[aid] < len(preferences[aid]):
                    free_agents.append(aid)

            if agent_id in accepted:
                matched = True
                break

        if not matched:
            continue

    return list(matching.items())
def execute_da_on_dataset(
    dataset: Dataset,
    precedence: Optional[List[Any]] = None,
) -> Outcome:
    """
    Dataset を使って論文の DA を実行
    """

    agents_obj, categories_obj = dataset.to_algorithm_inputs()

    matching = da_algorithm(agents_obj, categories_obj, precedence)

    return Outcome(
        dataset_id=dataset.id,
        algorithm_name="Reserve-Induced-DA",
        matching={a: c for a, c in matching},
    )
