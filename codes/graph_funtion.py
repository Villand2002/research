import networkx as nx
import numpy as np
from typing import List, Set
from codes.agent import Agents, Category
from codes.data_generation.mallows import generate_mallows_permutation

def assign_eligibility_random(agents_obj: Agents, categories_obj: List[Category], p=0.4):
    """Agents と Category の構造に合わせて eligibility を設定"""
    for cat in categories_obj:
        eligible = []
        for ag in agents_obj.agents:
            if np.random.rand() < p:
                eligible.append(ag.agent_id)
        if len(eligible) == 0:
            eligible.append(np.random.choice([ag.agent_id for ag in agents_obj.agents]))
        cat.eligible_agents = eligible

def assign_category_priorities(agents_obj: Agents, categories_obj: List[Category], phi=0.6):
    """各カテゴリに対して agent の優先順位を Mallows で生成し、Categoryオブジェクトにセット"""
    reference = [ag.agent_id for ag in agents_obj.agents]
    for cat in categories_obj:
        perm = generate_mallows_permutation(reference, phi)
        cat.priority = perm

def compute_max_matching_size(agents_obj: Agents, categories_obj: List[Category], banned_agents: Set[int]):
    """クラス構造に合わせた最大マッチング計算 (Max Flow)"""
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

def nx_rebuild_graph(agents_obj: Agents, categories_obj: List[Category], banned_agents: Set[int]):
    """最終マッチング構築用にグラフを再構築"""
    G = nx.DiGraph()
    S, T = "S", "T"
    G.add_node(S); G.add_node(T)

    for ag in agents_obj.agents:
        if ag.agent_id not in banned_agents:
            G.add_edge(S, ag.agent_id, capacity=1)

    for cat in categories_obj:
        rank = {a: i for i, a in enumerate(cat.priority)}
        for ag in cat.eligible_agents:
            if ag not in banned_agents:
                allow = True
                for r in banned_agents:
                    if r in rank and ag in rank and rank[r] < rank[ag]:
                        allow = False
                        break
                if allow:
                    G.add_edge(ag, cat.category_id, capacity=1)
        G.add_edge(cat.category_id, T, capacity=cat.capacity)

    return G