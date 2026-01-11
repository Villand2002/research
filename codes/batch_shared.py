from dataclasses import dataclass
from typing import Dict, List, Tuple

from codes.data_generation.dataset import Dataset


@dataclass(frozen=True)
class BatchDatasetConfig:
    num_agents: int = 35
    num_categories: int = 10
    capacity_ratio: float = 1.0
    capacity_std: float = 0.2
    eligibility_prob: float = 0.55
    priority_phi: float = 0.8
    preference_phi: float = 0.7
    max_preference_length: int = 5


DEFAULT_BATCH_CONFIG = BatchDatasetConfig()
BATCH_DATASET_SEEDS = tuple(range(100))


def build_batch_dataset(seed: int, config: BatchDatasetConfig = DEFAULT_BATCH_CONFIG) -> Dataset:
    return Dataset.build(
        dataset_id=seed,
        num_agents=config.num_agents,
        num_categories=config.num_categories,
        capacity_ratio=config.capacity_ratio,
        capacity_std=config.capacity_std,
        eligibility_prob=config.eligibility_prob,
        priority_phi=config.priority_phi,
        preference_phi=config.preference_phi,
        max_preference_length=config.max_preference_length,
        seed=seed,
    )


def build_scu_inputs_from_dataset(
    dataset: Dataset,
) -> Tuple[List[str], List[str], Dict[str, int], Dict[str, List[str]], List[str], List[str]]:
    agents = [f"i{ag.agent_id}" for ag in dataset.agents]
    categories = [f"c{cat.category_id}" for cat in dataset.categories]
    capacities = {f"c{cat.category_id}": cat.capacity for cat in dataset.categories}
    priorities = {
        f"c{cat.category_id}": [f"i{aid}" for aid in cat.priority] for cat in dataset.categories
    }
    precedence = categories[:]
    beneficial_cutoff = max(1, len(categories) // 2)
    beneficial = categories[:beneficial_cutoff]
    return agents, categories, capacities, priorities, precedence, beneficial
