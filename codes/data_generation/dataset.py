import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from codes.agent import Agent, Agents, Category
from codes.data_generation.mallows import generate_mallows_permutation


@dataclass
class Dataset:
    """Container for agents and categories used by matching algorithms."""

    id: int
    agents: List[Agent]
    categories: List[Category]

    @classmethod
    def build(
        cls,
        dataset_id: int,
        num_agents: int,
        num_categories: int,
        capacity_ratio: float = 1.0,
        capacity_std: float = 0.1,
        eligibility_prob: float = 0.5,
        priority_phi: float = 0.9,
        preference_phi: float = 0.7,
        max_preference_length: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> "Dataset":
        """
        Quick dataset factory. Generates Agents/Categories objects that are immediately
        usable by the matching algorithms.
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        agent_ids = list(range(num_agents))
        category_ids = list(range(num_categories))

        total_capacity = max(1, int(round(num_agents * capacity_ratio)))
        capacities = cls._draw_capacities(num_categories, total_capacity, capacity_std)

        categories = []
        for cid in category_ids:
            priority = generate_mallows_permutation(agent_ids, priority_phi)
            categories.append(Category(category_id=cid, capacity=capacities[cid], priority=priority))
            categories[-1].eligible_agents = []

        agents: List[Agent] = []
        max_pref = max(1, min(num_categories, max_preference_length or num_categories))
        for aid in agent_ids:
            ranking = generate_mallows_permutation(category_ids, preference_phi)

            acceptable: List[int] = []
            for cat_id in ranking:
                if random.random() <= eligibility_prob:
                    acceptable.append(cat_id)
                if len(acceptable) >= max_pref:
                    break
            if not acceptable:
                acceptable = ranking[:1]

            agents.append(Agent(agent_id=aid, acceptable_categories=acceptable))
            for cat_id in acceptable:
                categories[cat_id].eligible_agents.append(aid)

        dataset = cls(dataset_id, agents, categories)
        dataset._category_lookup = {cat.category_id: cat for cat in categories}
        return dataset

    @staticmethod
    def _draw_capacities(num_categories: int, total_capacity: int, std_ratio: float) -> List[int]:
        mean = total_capacity / float(num_categories)
        std = mean * std_ratio

        caps = np.random.normal(loc=mean, scale=std, size=num_categories)
        caps = np.maximum(caps, 1).astype(int)

        diff = int(total_capacity - caps.sum())
        while diff != 0:
            idx = np.random.randint(0, num_categories)
            if diff > 0:
                caps[idx] += 1
                diff -= 1
            elif caps[idx] > 1:
                caps[idx] -= 1
                diff += 1

        return caps.tolist()

    def to_algorithm_inputs(self) -> Tuple[Agents, List[Category]]:
        """Return objects consumable by the algorithms."""
        return Agents(self.agents, len(self.agents)), self.categories

    def get_category(self, category_id: int) -> Category:
        if not hasattr(self, "_category_lookup"):
            self._category_lookup = {cat.category_id: cat for cat in self.categories}
        return self._category_lookup[category_id]

    def get_agent(self, agent_id: int) -> Agent:
        if not hasattr(self, "_agent_lookup"):
            self._agent_lookup = {ag.agent_id: ag for ag in self.agents}
        return self._agent_lookup[agent_id]


if __name__ == "__main__":
    # Minimal example
    ds = Dataset.build(
        dataset_id=1,
        num_agents=5,
        num_categories=3,
        capacity_ratio=1.0,
        capacity_std=0.2,
        eligibility_prob=0.6,
        priority_phi=0.8,
        preference_phi=0.7,
        seed=42,
    )

    print("Agents (acceptable categories):")
    for ag in ds.agents:
        print(f"  Agent {ag.agent_id}: {ag.acceptable_categories}")

    print("\nCategories (capacity / priority top 3):")
    for cat in ds.categories:
        print(f"  Category {cat.category_id} cap={cat.capacity}, priority_head={cat.priority[:3]}")
