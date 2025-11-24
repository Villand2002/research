import networkx as nx
from typing import List, Tuple, Dict
from agent import Agents, Category
from graph_function import compute_max_matching_size, nx_rebuild_graph

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

def execute_mma(agents_obj: Agents, categories_obj: List[Category]) -> List[Tuple[int, int]]:
    """Maximum Matching Adjustment (MMA) Algorithm"""
    agents = agents_obj.agents
    agent_ids = [ag.agent_id for ag in agents]

    # Step 1: Eligibility Graph G を構築
    G = nx.Graph()
    for ag in agents:
        for cat in categories_obj:
            if ag.agent_id in cat.eligible_agents:
                G.add_edge(f"A_{ag.agent_id}", f"C_{cat.category_id}")

    # Step 2: 最大マッチング μ を求める
    max_matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    # μ の内部表現 {agent -> category}
    mu: Dict[int, int] = {}
    for a, c in max_matching:
        if a.startswith("A_"):
            agent = int(a[2:])
            category = int(c[2:])
        else:
            agent = int(c[2:])
            category = int(a[2:])
        mu[agent] = category

    # Step 3: Unmatched agent の調整
    unmatched_agents = [ag for ag in agent_ids if ag not in mu]

    for i in unmatched_agents:
        cats = [cat for cat in categories_obj if i in cat.eligible_agents]

        for c in cats:
            cat_id = c.category_id
            matched_agents = [a for a, cat in mu.items() if cat == cat_id]

            if len(matched_agents) < c.capacity:
                mu[i] = cat_id
                break

            priorities = c.priority
            # 最も優先順位が低いエージェントを探す（indexが大きいほど優先度低）
            # 注意: matched_agentsの中にpriorityリストにないエージェントがいるとエラーになるためチェック推奨
            valid_matched = [ma for ma in matched_agents if ma in priorities]
            if not valid_matched:
                continue
                
            i_prime = max(valid_matched, key=lambda a: priorities.index(a))

            if i in priorities and priorities.index(i) < priorities.index(i_prime):
                mu[i] = cat_id
                del mu[i_prime]
                break

    return list(mu.items())