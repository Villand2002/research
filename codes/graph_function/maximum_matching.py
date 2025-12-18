from typing import List, Set

import networkx as nx
from codes.agent import Agents, Category


def compute_max_matching_size(
    agents_obj: Agents, 
    categories_obj: List[Category], 
    already_banned: Set[int], 
    target_ag_id: int = None
):
    """
    論文の G^i (修正適格グラフ) を構築し、最大流を計算する内部関数 
    """
    G = nx.DiGraph()
    S, T = "S", "T"
    
    # 判定から除外されるエージェントの集合
    exclude = already_banned.copy()
    if target_ag_id is not None:
        exclude.add(target_ag_id)

    # ソースからエージェントへのエッジ
    for ag in agents_obj.agents:
        if ag.agent_id not in exclude:
            G.add_edge(S, ag.agent_id, capacity=1)

    # エージェントからカテゴリーへのエッジ (優先順位による制限付き)
    for cat in categories_obj:
        G.add_edge(cat.category_id, T, capacity=cat.capacity)
        for eligible_ag_id in cat.eligible_agents:
            # すでに拒絶された、または現在テスト中のエージェントはスキップ
            if eligible_ag_id in exclude:
                continue
            
            # 優先順位の尊重ロジック[cite: 116, 175]:
            # もし target_ag_id (i) が存在し、そのカテゴリー c において
            # i >_c eligible_ag_id (j) であるなら、エッジ (j, c) は G^i に含めない
            if target_ag_id is not None:
                if target_ag_id in cat.priority and eligible_ag_id in cat.priority:
                    # indexが小さいほど優先順位が高い
                    if cat.priority.index(target_ag_id) < cat.priority.index(eligible_ag_id):
                        continue
            
            G.add_edge(eligible_ag_id, cat.category_id, capacity=1)

    flow_val, _ = nx.maximum_flow(G, S, T)
    return flow_val
