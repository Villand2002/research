import numpy as np
from dataclasses import dataclass
import random

# --------------------------
# Agent and Category
# --------------------------

@dataclass
class Agent:
    id: int
    preferences: list  # list of acceptable category ids (subset)


@dataclass
class Category:
    id: int
    priority: list  # list of agent ids in order of priority
    capacity: int


# --------------------------
# Mallows (RIM version)
# See Definition 2: https://icml.cc/2011/papers/135_icmlpaper.pdf
# --------------------------

def sample_mallows_rim(reference, phi):
    """
    Sample a ranking from the Mallows model using the repeated insertion method (RIM).

    Args:
        reference (list): reference ranking (canonical order)
        phi (float): dispersion parameter (0 < phi <= 1)
    
    Returns:
        ranking (list): a sampled ranking
    """
    m = len(reference)
    ranking = []
    for i in range(m):
        size = i + 1
        # Ensure probs are floats
        probs = np.array([phi**(size - j - 1) for j in range(size)], dtype=float)
        probs /= probs.sum()  # normalize to sum=1
        k = np.random.choice(size, p=probs)
        ranking.insert(k, reference[i])
    return ranking


# --------------------------
# Dataset class
# --------------------------

class Dataset:
    def __init__(self, dataset_id,
                 num_agents,
                 num_categories,
                 category_capacity_ratio=1.0,
                 category_capacity_std=0.1,
                 indifference_subset_size=None,
                 category_phi=0.9,
                 seed=None):

        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.id = dataset_id
        self.num_agents = num_agents
        self.num_categories = num_categories
        self.category_capacity_ratio = category_capacity_ratio
        self.category_capacity_std = category_capacity_std
        self.indifference_subset_size = indifference_subset_size or num_categories
        self.category_phi = category_phi  # Mallows parameter for category priorities

        self.agents = []
        self.categories = []

    # -------------------
    # Generate agents
    # -------------------
    def generate_agents(self):
        """
        Generate agents with preferences.
        Each agent is indifferent among a random subset of categories.
        """
        for aid in range(self.num_agents):
            prefs = random.sample(
                range(self.num_categories),
                k=min(self.indifference_subset_size, self.num_categories)
            )
            self.agents.append(Agent(aid, prefs))

    # -------------------
    # Generate categories
    # -------------------
    def generate_categories(self):
        # Step 1: capacities based on normal distribution
        total_capacity = int(self.num_agents * self.category_capacity_ratio)
        mean = total_capacity / float(self.num_categories)  # ensure float division
        std = mean * self.category_capacity_std

        # Generate capacities as float numbers
        caps = np.random.normal(loc=mean, scale=std, size=self.num_categories)
        caps = np.maximum(caps, 1)   # ensure minimum capacity is 1

        # Convert to integers
        caps = caps.astype(int)

        # Adjust capacities so that total matches exactly total_capacity
        diff = total_capacity - caps.sum()
        while diff != 0:
            i = np.random.randint(0, self.num_categories)
            if diff > 0:
                caps[i] += 1
                diff -= 1
            else:
                if caps[i] > 1:  # never reduce below 1
                    caps[i] -= 1
                    diff += 1

        # Step 2: generate priority lists for each category
        # Using Mallows (RIM) instead of random shuffle
        reference = list(range(self.num_agents))
        for cid in range(self.num_categories):
            priority = sample_mallows_rim(reference, self.category_phi)
            self.categories.append(Category(cid, priority, caps[cid]))

    # -------------------
    # All-in-one generate
    # -------------------
    def generate(self):
        self.generate_agents()
        self.generate_categories()
        return self


# --------------------------
# Example usage
# --------------------------

if __name__ == "__main__":
    # Create a dataset with 10 agents and 5 categories
    dataset = Dataset(
        dataset_id=1,
        num_agents=10,
        num_categories=5,
        category_capacity_ratio=1.2,   # total capacity = 1.2 * num_agents
        category_capacity_std=0.1,     # 10% std for capacities
        indifference_subset_size=3,    # each agent indifferent among 3 categories
        category_phi=0.8,              # Mallows dispersion for category priority
        seed=42                         # for reproducibility
    ).generate()

    # Print the agents and their acceptable categories
    print("Agents and their acceptable categories:")
    for agent in dataset.agents:
        print(f"Agent {agent.id}: {agent.preferences}")

    # Print the categories, their capacities, and Mallows-based priority over agents
    print("\nCategories, capacities, and priority over agents:")
    for category in dataset.categories:
        print(f"Category {category.id}: capacity={category.capacity}, priority={category.priority}")



    # create 100 datasets
    datasets = []
    for i in range(100):
        ds = Dataset(i, 1000, 50, 5, 0.7, 0.8, 1.2, 0.1, 0.8).generate()
        datasets.append(ds)