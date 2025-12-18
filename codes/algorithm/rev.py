from typing import List

import networkx as nx

from codes.agent import Agents, Category, Outcome
from codes.data_generation.dataset import Dataset
from codes.graph_function.flow_network import nx_rebuild_graph
from codes.graph_function.maximum_matching import compute_max_matching_size


def rev_algorithm(agents_obj: Agents, categories_obj: List[Category]):
    """REV Algorithm implementation"""
    agents_list = [ag.agent_id for ag in agents_obj.agents]
    banned = set()

    base_val = compute_max_matching_size(agents_obj, categories_obj, banned)

    # baseline は agent_id の昇順を弱者側とみなす（降順でチェック）
    for ag in reversed(agents_list):
        test = banned | {ag}
        val = compute_max_matching_size(agents_obj, categories_obj, test)
        if val == base_val:
            banned = test

    flow_value, flow_dict = nx.maximum_flow(
        nx_rebuild_graph(agents_obj, categories_obj, banned), "S", "T"
    )

    matching = []
    for ag in agents_list:
        if ag in flow_dict:
            for cat_id, flow in flow_dict[ag].items():
                if cat_id == "T":
                    continue
                if flow > 0:
                    matching.append((ag, cat_id))

    return matching, banned


def execute_rev_on_dataset(dataset: Dataset) -> Outcome:
    """Run REV on a Dataset and return an Outcome."""
    agents_obj, categories_obj = dataset.to_algorithm_inputs()
    matching, _ = rev_algorithm(agents_obj, categories_obj)
    matching_dict = {a: c for a, c in matching}
    return Outcome(dataset_id=dataset.id, algorithm_name="REV", matching=matching_dict)


# Alias for VEV naming in the notes
def execute_vev_on_dataset(dataset: Dataset) -> Outcome:
    return execute_rev_on_dataset(dataset)
