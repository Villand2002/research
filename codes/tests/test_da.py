import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.agent import Agent, Agents, Category
from codes.algorithm.da import da_algorithm, execute_da_on_dataset
from codes.data_generation.dataset import Dataset


class TestDAAlgorithm:
    def test_da_example1_like(self):
        a1 = Agent(agent_id=1, acceptable_categories=[1])
        a2 = Agent(agent_id=2, acceptable_categories=[1, 2])
        a3 = Agent(agent_id=3, acceptable_categories=[2])
        agents_obj = Agents(agents=[a1, a2, a3], agent_number=3)

        c1 = Category(category_id=1, capacity=1, priority=[2, 1])
        c2 = Category(category_id=2, capacity=1, priority=[2, 3])
        c1.eligible_agents = [2, 1]
        c2.eligible_agents = [2, 3]

        matching = dict(da_algorithm(agents_obj, [c1, c2]))
        assert matching == {2: 1, 3: 2}

    def test_execute_da_on_dataset_feasible_and_stable(self):
        dataset = Dataset.build(
            dataset_id=101,
            num_agents=12,
            num_categories=4,
            capacity_ratio=1.0,
            capacity_std=0.0,
            eligibility_prob=0.6,
            priority_phi=0.8,
            preference_phi=0.7,
            seed=7,
        )
        outcome = execute_da_on_dataset(dataset)
        feasible, violations = outcome.verify_feasible(dataset)
        assert feasible, f"DA produced infeasible matching: {violations} violations"
