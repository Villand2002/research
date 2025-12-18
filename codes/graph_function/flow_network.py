from typing import List, Set

import networkx as nx
from codes.agent import Agents, Category


def nx_rebuild_graph(agents_obj: Agents, categories_obj: List[Category], banned_agents: Set[int]):
    """Rebuild flow network with banned agents filtered out."""
    G = nx.DiGraph()
    S, T = "S", "T"
    G.add_node(S)
    G.add_node(T)

    for ag in agents_obj.agents:
        if ag.agent_id not in banned_agents:
            G.add_edge(S, ag.agent_id, capacity=1)

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
        G.add_edge(cat.category_id, T, capacity=cat.capacity)

    return G
