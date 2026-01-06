import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.agent import Agent, Agents, Category
from codes.algorithm.rev_bipartite import rev_algorithm


class TestREVBipartiteAlgorithm:
    @pytest.fixture
    def setup_example1_objects(self):
        """
        [cite_start]論文 Example 1 [cite: 94-102] のデータセットアップ

        Agents: {1, 2, 3}
        Categories: {1, 2} (Capacity 1)
        Priorities:
            c1: 2 > 1  (3は適格外)
            c2: 2 > 3  (1は適格外)
        """
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

    def test_rev_bipartite_example1(self, setup_example1_objects):
        agents_obj, categories_obj = setup_example1_objects

        matching, banned = rev_algorithm(agents_obj, categories_obj)

        matching_dict = {m[0]: m[1] for m in matching}

        assert len(matching) == 2
        assert 2 in matching_dict
        assert len(banned) == 1
