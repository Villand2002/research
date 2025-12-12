from dataset import *
from dataclasses import dataclass, field

# --------------------------
# Outcome class
# --------------------------

@dataclass
class Outcome:
    dataset_id: int
    algorithm_name: str
    matching: dict = field(default_factory=dict)  # {agent_id: category_id or None}
    
    def verify_fairness(self, dataset):
        """
        Check for fairness: no agent prefers another agent's assigned category
        and has higher priority.
        
        Returns:
            is_fair (bool): True if no violations found, False otherwise
            envy_count (int): number of fairness violations (justified envy)
        """
        if dataset is None:
            raise ValueError("Dataset must be provided for fairness check.")

        envy_count = 0

        for agent_id, assigned_cat in self.matching.items():
            agent = dataset.agents[agent_id]
            for cat_id in agent.preferences:
                if cat_id == assigned_cat:
                    break
                category = dataset.categories[cat_id]
                for other_agent_id, other_assigned_cat in self.matching.items():
                    if other_assigned_cat == cat_id:
                        if category.priority.index(agent_id) < category.priority.index(other_agent_id):
                            envy_count += 1

        is_fair = envy_count == 0
        return is_fair, envy_count

    def verify_stability(self, dataset):
        raise NotImplementedError("Stability check not implemented yet.")

    def verify_nonwasteful(self, dataset):
        raise NotImplementedError("Nonwastefulness check not implemented yet.")

# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    # Step 1: create dataset
    dataset = Dataset(
        dataset_id=1,
        num_agents=3,
        num_categories=2,
        category_capacity_ratio=1,     # total capacity = num_agents * ratio
        category_capacity_std=0.1,    # 10% std for capacities
        indifference_subset_size=1,   # each agent indifferent among 1 category
        category_phi=1,               # Mallows dispersion for category priority
        seed=42
    ).generate()

    # Step 2: create an example matching
    # Suppose agent 0 -> category 1, agent 1 -> category 0, agent 2 -> category 1
    matching_result = {0: 1, 1: 0, 2: 1}

    # Step 3: create Outcome object
    outcome = Outcome(
        dataset_id=1,
        algorithm_name="ExampleMatching",
        matching=matching_result
    )

    # Step 4: check fairness
    is_fair, envy_count = outcome.verify_fairness(dataset)
    print(f"Is the matching fair? {is_fair}")
    print(f"Number of envy cases: {envy_count}")