from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Tuple

if TYPE_CHECKING:
    from codes.data_generation.dataset import Dataset

class Agent:
    def __init__(self, agent_id: int, acceptable_categories: List[int]):
        self.agent_id = agent_id
        self.acceptable_categories = acceptable_categories

    def show_info(self):
        print(f"Agent ID: {self.agent_id}, Acceptable Categories: {self.acceptable_categories}")

class Agents:
    def __init__(self, agents: List[Agent], agent_number: int):
        self.agents = agents
        self.agent_number = agent_number

    def show_all_info(self):
        for agent in self.agents:
            agent.show_info()

class Category:
    def __init__(self, category_id: int, capacity: int, priority: List[int]):
        self.category_id = category_id
        self.capacity = capacity
        self.priority = priority
        # アルゴリズムで使用する動的プロパティ
        self.eligible_agents: List[int] = [] 

    def show_info(self):
        print(f"Category ID: {self.category_id}, Capacity: {self.capacity}, Priority: {self.priority}")


@dataclass
class Outcome:
    dataset_id: int
    algorithm_name: str
    matching: Dict[int, int] = field(default_factory=dict)  # {agent_id: category_id or None}

    def _category_allocations(self, dataset: "Dataset") -> Dict[int, List[int]]:
        allocations = {cat.category_id: [] for cat in dataset.categories}
        for agent_id, cat_id in self.matching.items():
            if cat_id is None:
                continue
            allocations.setdefault(cat_id, []).append(agent_id)
        return allocations

    @staticmethod
    def _priority_ranks(dataset: "Dataset") -> Dict[int, Dict[int, int]]:
        return {cat.category_id: {aid: idx for idx, aid in enumerate(cat.priority)} for cat in dataset.categories}

    def verify_feasible(self, dataset: "Dataset") -> Tuple[bool, int]:
        """
        Check basic feasibility: capacity, eligibility, and acceptable categories.
        Returns (is_feasible, violation_count).
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for feasibility check.")

        violations = 0
        allocations = self._category_allocations(dataset)

        for agent_id, cat_id in self.matching.items():
            if cat_id is None:
                continue
            try:
                agent = dataset.get_agent(agent_id)
                category = dataset.get_category(cat_id)
            except KeyError:
                violations += 1
                continue

            if cat_id not in agent.acceptable_categories:
                violations += 1
            if agent_id not in category.eligible_agents:
                violations += 1

        for cat_id, assigned in allocations.items():
            category = dataset.get_category(cat_id)
            overflow = len(assigned) - category.capacity
            if overflow > 0:
                violations += overflow

        return violations == 0, violations

    def verify_fairness(self, dataset: "Dataset") -> Tuple[bool, int]:
        """
        Check fairness (justified envy):
        - matched agents are treated as non-envious.
        - for each unmatched agent i and each eligible category c,
          ensure there is no matched agent j with lower priority than i in c.
        Returns (is_fair, envy_count).
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for fairness check.")

        allocations = self._category_allocations(dataset)
        priority_rank = self._priority_ranks(dataset)
        envy_count = 0

        for agent in dataset.agents:
            assigned_cat = self.matching.get(agent.agent_id)
            if assigned_cat is not None:
                continue
            for cat_id in agent.acceptable_categories:
                
                category = dataset.get_category(cat_id)
                if agent.agent_id not in category.eligible_agents:
                    continue
                if agent.agent_id not in priority_rank.get(cat_id, {}):
                    continue
                for other_agent in allocations.get(cat_id, []):
                    # priorityが低いのに割り当てられているかの確認
                    if priority_rank[cat_id].get(agent.agent_id, float("inf")) < priority_rank[cat_id].get(other_agent, float("inf")):
                        envy_count += 1
                        break

        return envy_count == 0, envy_count

    def verify_nonwasteful(self, dataset: "Dataset") -> Tuple[bool, int]:
        """
        Check non-wastefulness: there is no unmatched agent who finds a category
        acceptable and for which capacity remains.
        Returns (is_nonwasteful, wasted_slots_with_demand).
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for nonwastefulness check.")

        allocations = self._category_allocations(dataset)
        wasted = 0

        unmatched = {ag.agent_id for ag in dataset.agents if self.matching.get(ag.agent_id) is None}
        for cat_id, assigned in allocations.items():
            category = dataset.get_category(cat_id)
            free_slots = max(category.capacity - len(assigned), 0)
            if free_slots == 0:
                continue
            for agent_id in list(unmatched):
                agent = dataset.get_agent(agent_id)
                if cat_id in agent.acceptable_categories and agent_id in category.eligible_agents:
                    wasted += 1
                    break

        return wasted == 0, wasted

    def verify_stability(self, dataset: "Dataset") -> Tuple[bool, int]:
        """
        Check stability: no blocking pair (agent, category).
        A blocking pair exists if an agent prefers a category to their assignment,
        is eligible for it, and the category has room or would prefer the agent
        over its worst current assignee.
        Returns (is_stable, blocking_pairs).
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for stability check.")

        allocations = self._category_allocations(dataset)
        priority_rank = self._priority_ranks(dataset)
        blocking_pairs = 0

        for agent in dataset.agents:
            assigned_cat = self.matching.get(agent.agent_id)
            for cat_id in agent.acceptable_categories:
                if cat_id == assigned_cat:
                    break
                category = dataset.get_category(cat_id)
                if agent.agent_id not in category.eligible_agents:
                    continue

                current = allocations.get(cat_id, [])
                if len(current) < category.capacity:
                    blocking_pairs += 1
                    break

                worst = max(current, key=lambda a: priority_rank.get(cat_id, {}).get(a, float("inf")))
                if priority_rank.get(cat_id, {}).get(agent.agent_id, float("inf")) < priority_rank.get(cat_id, {}).get(worst, float("inf")):
                    blocking_pairs += 1
                    break

        return blocking_pairs == 0, blocking_pairs
