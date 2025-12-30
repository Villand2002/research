import networkx as nx
from typing import List, Set, Tuple, Dict, Any, Optional
from codes.agent import Agents, Category, Outcome
from codes.data_generation.dataset import Dataset


def rev_algorithm_flow_network(
    agents_obj: Agents, 
    categories_obj: List[Category], 
    baseline_order: Optional[List[Any]] = None
) -> Tuple[List[Tuple[Any, Any]], Set[Any]]:
    """
    Reverse Rejecting (REV) Rule の実装 (flow network)
    
    論文: Algorithm 2 (p.6)
    
    Input: A reserve system R = (I, C, ≻, q) and a baseline order π over the agents I
    Output: A matching µ
    
    Parameters:
    -----------
    agents_obj : Agents
        エージェントの集合
    categories_obj : List[Category]
        カテゴリのリスト
    baseline_order : Optional[List[Any]]
        エージェントのベースライン順序 π（指定がない場合はagent_idの順序を使用）
    
    Returns:
    --------
    matching : List[Tuple[Any, Any]]
        マッチング結果 [(agent_id, category_id), ...]
    rejected : Set[Any]
        拒絶されたエージェントの集合
    """
    
    # エージェントIDのリストを取得
    agents_list: List[Any] = [ag.agent_id for ag in agents_obj.agents]
    
    # カテゴリIDの集合 (デバッグや検証用途)
    # category_ids = {cat.category_id for cat in categories_obj}
    
    # ベースライン順序 π の設定（指定がない場合はagents_listをそのまま使用）
    if baseline_order is None:
        baseline_order = agents_list.copy()
    
    # ノードIDの衝突を避けるためのプレフィックス
    # エージェントIDとカテゴリIDが同じ値の場合に区別するため
    def agent_node(aid: Any) -> str:
        return f"agent_{aid}"
    
    def category_node(cid: Any) -> str:
        return f"category_{cid}"
    
    def is_eligible(agent_id: Any, category: Category) -> bool:
        """エージェントがカテゴリに eligible かどうかを判定"""
        return agent_id in category.eligible_agents
    
    def has_higher_priority(agent_i: Any, agent_j: Any, category: Category) -> bool:
        """
        カテゴリ c において agent_i が agent_j より優先されるかを判定
        i ≻_c j が成り立つか
        
        priority リストは先頭が最優先（index が小さいほど優先度が高い）
        """
        if agent_i not in category.priority or agent_j not in category.priority:
            return False
        idx_i = category.priority.index(agent_i)
        idx_j = category.priority.index(agent_j)
        return idx_i < idx_j
    
    def get_max_matching_size(
        rejected_agents: Set[Any],
        current_agent_i: Optional[Any] = None,
        removed_edges_from_banned: Optional[Dict[Any, Set[Any]]] = None
    ) -> int:
        """
        指定された制約下での最大マッチングサイズを計算
        
        論文の G^i グラフを構築:
        1. rejected_agents（すでに拒絶されたエージェント）を除外
        2. current_agent_i（現在判定中のエージェント）を除外
        3. i ≻_c j となるエッジ (j, c) を除外（iがjより優先される場合）
        4. すでにbannedされたエージェントによって削除されたエッジも除外
        """
        if removed_edges_from_banned is None:
            removed_edges_from_banned = {}
        
        # フローネットワークの構築
        G = nx.DiGraph()
        SOURCE = "SOURCE_NODE"
        SINK = "SINK_NODE"
        
        # アクティブなエージェント（rejected でも current_agent_i でもない）
        active_agents: List[Any] = [
            aid for aid in agents_list 
            if aid not in rejected_agents and aid != current_agent_i
        ]
        
        # ソースからエージェントへのエッジ（容量1）
        for aid in active_agents:
            G.add_edge(SOURCE, agent_node(aid), capacity=1)
        
        # カテゴリからシンクへのエッジ、およびエージェントからカテゴリへのエッジ
        for cat in categories_obj:
            cid = cat.category_id
            
            # カテゴリからシンクへのエッジ（容量 = カテゴリの容量）
            G.add_edge(category_node(cid), SINK, capacity=cat.capacity)
            
            # eligible なエージェントからカテゴリへのエッジ
            for agent_j in cat.eligible_agents:
                # rejected または current_agent_i なら除外
                if agent_j in rejected_agents or agent_j == current_agent_i:
                    continue
                
                # bannedされたエージェントによって削除されたエッジなら除外
                if cid in removed_edges_from_banned and agent_j in removed_edges_from_banned[cid]:
                    continue
                
                # current_agent_i が存在し、i ≻_c j なら、エッジ (j, c) を除外
                if current_agent_i is not None:
                    if is_eligible(current_agent_i, cat):
                        if has_higher_priority(current_agent_i, agent_j, cat):
                            continue
                
                # エッジを追加
                G.add_edge(agent_node(agent_j), category_node(cid), capacity=1)
        
        # ソースまたはシンクがグラフにない場合は0を返す
        if SOURCE not in G or SINK not in G:
            return 0
        
        # 最大フローを計算
        flow_value, _ = nx.maximum_flow(G, SOURCE, SINK)
        return int(flow_value)
    
    def compute_final_matching(
        rejected_agents: Set[Any],
        removed_edges: Dict[Any, Set[Any]]
    ) -> List[Tuple[Any, Any]]:
        """
        最終的なマッチングを計算
        """
        G = nx.DiGraph()
        SOURCE = "SOURCE_NODE"
        SINK = "SINK_NODE"
        
        # rejected でないエージェント
        active_agents = [aid for aid in agents_list if aid not in rejected_agents]
        
        for aid in active_agents:
            G.add_edge(SOURCE, agent_node(aid), capacity=1)
        
        for cat in categories_obj:
            cid = cat.category_id
            G.add_edge(category_node(cid), SINK, capacity=cat.capacity)
            
            for agent_j in cat.eligible_agents:
                if agent_j in rejected_agents:
                    continue
                if cid in removed_edges and agent_j in removed_edges[cid]:
                    continue
                G.add_edge(agent_node(agent_j), category_node(cid), capacity=1)
        
        if SOURCE not in G or SINK not in G:
            return []
        
        _, flow_dict = nx.maximum_flow(G, SOURCE, SINK)
        
        # マッチングを抽出
        matching: List[Tuple[Any, Any]] = []
        for aid in active_agents:
            anode = agent_node(aid)
            if anode in flow_dict:
                for target, flow in flow_dict[anode].items():
                    # targetがカテゴリノードかどうかをチェック
                    if isinstance(target, str) and target.startswith("category_") and flow > 0:
                        # カテゴリIDを抽出
                        cid_str = target[len("category_"):]
                        # 元のカテゴリIDの型に戻す
                        for cat in categories_obj:
                            if str(cat.category_id) == cid_str:
                                matching.append((aid, cat.category_id))
                                break
        
        return matching
    
    # ========================================
    # REV アルゴリズム本体
    # ========================================
    
    # Step 1: 初期状態での最大マッチングサイズ m を計算
    base_m: int = get_max_matching_size(rejected_agents=set())
    
    # 拒絶されたエージェントの集合 R
    rejected: Set[Any] = set()
    
    # bannedされたエージェントによって削除されたエッジを追跡
    # {category_id: {agent_id, ...}, ...}
    removed_edges: Dict[Any, Set[Any]] = {}
    
    # Step 2: ベースライン順序 π の逆順で各エージェントを処理
    for agent_i in reversed(baseline_order):
        # G^i を構築して最大マッチングサイズを計算
        # G^i: agent_i を除外し、i ≻_c j となるエッジ (j, c) を除外
        matching_size = get_max_matching_size(
            rejected_agents=rejected,
            current_agent_i=agent_i,
            removed_edges_from_banned=removed_edges
        )
        
        # もし G^i で最大マッチングサイズ m が維持できるなら、agent_i を拒絶
        if matching_size == base_m:
            # agent_i を rejected に追加
            rejected.add(agent_i)
            
            # グラフ G を更新: agent_i を除去し、i ≻_c j となるエッジ (j, c) を除去
            for cat in categories_obj:
                cid = cat.category_id
                if is_eligible(agent_i, cat):
                    # agent_i より優先度が低いエージェントのエッジを削除
                    for agent_j in cat.eligible_agents:
                        if agent_j != agent_i and has_higher_priority(agent_i, agent_j, cat):
                            if cid not in removed_edges:
                                removed_edges[cid] = set()
                            removed_edges[cid].add(agent_j)
    
    # Step 3: 最終的なマッチングを計算
    matching = compute_final_matching(rejected, removed_edges)
    
    return matching, rejected


def rev_algorithm_bipartite(
    agents_obj: Agents,
    categories_obj: List[Category],
    baseline_order: Optional[List[Any]] = None
) -> Tuple[List[Tuple[Any, Any]], Set[Any]]:
    """
    Reverse Rejecting (REV) Rule の実装 (bipartite graph)

    論文: Algorithm 2 (p.6)
    """
    agents_list: List[Any] = [ag.agent_id for ag in agents_obj.agents]

    if baseline_order is None:
        baseline_order = agents_list.copy()

    def is_eligible(agent_id: Any, category: Category) -> bool:
        return agent_id in category.eligible_agents

    def has_higher_priority(agent_i: Any, agent_j: Any, category: Category) -> bool:
        if agent_i not in category.priority or agent_j not in category.priority:
            return False
        idx_i = category.priority.index(agent_i)
        idx_j = category.priority.index(agent_j)
        return idx_i < idx_j

    def build_bipartite_graph(
        rejected_agents: Set[Any],
        current_agent_i: Optional[Any] = None,
        removed_edges_from_banned: Optional[Dict[Any, Set[Any]]] = None
    ) -> Tuple[nx.Graph, List[Tuple[str, Any]]]:
        if removed_edges_from_banned is None:
            removed_edges_from_banned = {}

        B = nx.Graph()
        left_nodes: List[Tuple[str, Any]] = []

        active_agents: List[Any] = [
            aid for aid in agents_list
            if aid not in rejected_agents and aid != current_agent_i
        ]

        for aid in active_agents:
            node = ("agent", aid)
            B.add_node(node, bipartite=0)
            left_nodes.append(node)

        for cat in categories_obj:
            for slot_idx in range(cat.capacity):
                slot_node = ("slot", cat.category_id, slot_idx)
                B.add_node(slot_node, bipartite=1)

        for cat in categories_obj:
            cid = cat.category_id
            for agent_j in cat.eligible_agents:
                if agent_j in rejected_agents or agent_j == current_agent_i:
                    continue
                if cid in removed_edges_from_banned and agent_j in removed_edges_from_banned[cid]:
                    continue
                if current_agent_i is not None and is_eligible(current_agent_i, cat):
                    if has_higher_priority(current_agent_i, agent_j, cat):
                        continue
                agent_node = ("agent", agent_j)
                if agent_node not in B:
                    continue
                for slot_idx in range(cat.capacity):
                    B.add_edge(agent_node, ("slot", cid, slot_idx))

        return B, left_nodes

    def get_max_matching_size(
        rejected_agents: Set[Any],
        current_agent_i: Optional[Any] = None,
        removed_edges_from_banned: Optional[Dict[Any, Set[Any]]] = None
    ) -> int:
        B, left_nodes = build_bipartite_graph(
            rejected_agents=rejected_agents,
            current_agent_i=current_agent_i,
            removed_edges_from_banned=removed_edges_from_banned,
        )
        if not left_nodes:
            return 0
        matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=left_nodes)
        return sum(1 for node in left_nodes if node in matching)

    def compute_final_matching(
        rejected_agents: Set[Any],
        removed_edges: Dict[Any, Set[Any]]
    ) -> List[Tuple[Any, Any]]:
        B, left_nodes = build_bipartite_graph(
            rejected_agents=rejected_agents,
            current_agent_i=None,
            removed_edges_from_banned=removed_edges,
        )
        if not left_nodes:
            return []
        matching = nx.algorithms.bipartite.maximum_matching(B, top_nodes=left_nodes)
        results: List[Tuple[Any, Any]] = []
        for node in left_nodes:
            partner = matching.get(node)
            if not partner or partner[0] != "slot":
                continue
            _, cid, _ = partner
            results.append((node[1], cid))
        return results

    base_m: int = get_max_matching_size(rejected_agents=set())
    rejected: Set[Any] = set()
    removed_edges: Dict[Any, Set[Any]] = {}

    for agent_i in reversed(baseline_order):
        matching_size = get_max_matching_size(
            rejected_agents=rejected,
            current_agent_i=agent_i,
            removed_edges_from_banned=removed_edges
        )

        if matching_size == base_m:
            rejected.add(agent_i)
            for cat in categories_obj:
                cid = cat.category_id
                if is_eligible(agent_i, cat):
                    for agent_j in cat.eligible_agents:
                        if agent_j != agent_i and has_higher_priority(agent_i, agent_j, cat):
                            if cid not in removed_edges:
                                removed_edges[cid] = set()
                            removed_edges[cid].add(agent_j)

    matching = compute_final_matching(rejected, removed_edges)
    return matching, rejected


def rev_algorithm(
    agents_obj: Agents,
    categories_obj: List[Category],
    baseline_order: Optional[List[Any]] = None
) -> Tuple[List[Tuple[Any, Any]], Set[Any]]:
    """
    Backward-compatible wrapper for the flow-network implementation.
    """
    return rev_algorithm_flow_network(agents_obj, categories_obj, baseline_order)


def execute_rev_on_dataset(dataset: Dataset) -> Outcome:
    """
    データセットに対してREVアルゴリズムを実行するエントリポイント
    """
    agents_obj, categories_obj = dataset.to_algorithm_inputs()
    matching, _ = rev_algorithm_flow_network(agents_obj, categories_obj)
    return Outcome(
        dataset_id=dataset.id,
        algorithm_name="REV with flow_network",
        matching={agent: cat for agent, cat in matching}
    )


def execute_rev_on_dataset_bipartite(dataset: Dataset) -> Outcome:
    """
    データセットに対してREVアルゴリズム (bipartite graph) を実行するエントリポイント
    """
    agents_obj, categories_obj = dataset.to_algorithm_inputs()
    matching, _ = rev_algorithm_bipartite(agents_obj, categories_obj)
    return Outcome(
        dataset_id=dataset.id,
        algorithm_name="REV with bipartite_graph",
        matching={agent: cat for agent, cat in matching}
    )
