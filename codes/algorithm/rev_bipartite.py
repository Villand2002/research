from typing import List, Set, Tuple, Any, Optional

from codes.agent import Agents, Category, Outcome
from codes.algorithm.rev import rev_algorithm_bipartite
from codes.data_generation.dataset import Dataset


def rev_algorithm(
    agents_obj: Agents,
    categories_obj: List[Category],
    baseline_order: Optional[List[Any]] = None
) -> Tuple[List[Tuple[Any, Any]], Set[Any]]:
    """
    Wrapper for the bipartite-graph REV implementation.
    """
    return rev_algorithm_bipartite(agents_obj, categories_obj, baseline_order)


def execute_rev_on_dataset(dataset: Dataset) -> Outcome:
    """
    Run REV (bipartite graph) using a Dataset and return an Outcome.
    """
    agents_obj, categories_obj = dataset.to_algorithm_inputs()
    matching, _ = rev_algorithm_bipartite(agents_obj, categories_obj)
    return Outcome(
        dataset_id=dataset.id,
        algorithm_name="REV with bipartite_graph",
        matching={agent: cat for agent, cat in matching},
    )
