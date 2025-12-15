import networkx as nx
from typing import List, Set

from codes.agent import Agents, Category


def compute_max_matching_size(agents_obj: Agents, categories_obj: List[Category], banned_agents: Set[int]):
    """Max matching size helper used by REV."""
    G = nx.DiGraph()
    S, T = "S", "T"
    G.add_node(S)
    G.add_node(T)

    for ag in agents_obj.agents:
        if ag.agent_id not in banned_agents:
            G.add_edge(S, ag.agent_id, capacity=1)

    for cat in categories_obj:
        G.add_edge(cat.category_id, T, capacity=cat.capacity)

    for cat in categories_obj:
        rank = {a: i for i, a in enumerate(cat.priority)}
        for ag in cat.eligible_agents:
            if ag in banned_agents:
                continue
            allow = True
            for r in banned_agents:
                if r in rank and ag in rank and rank[r] < rank[ag]:
                    allow = False
                    break
            if allow:
                G.add_edge(ag, cat.category_id, capacity=1)

    return nx.maximum_flow(G, S, T)[0]
