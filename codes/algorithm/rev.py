import networkx as nx
from typing import List, Tuple, Dict
from codes.agent import Agents, Category
from codes.graph_funtion import compute_max_matching_size, nx_rebuild_graph

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
                if flow > 0:
                    matching.append((ag, cat_id))

    return matching, banned