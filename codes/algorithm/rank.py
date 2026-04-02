import networkx as nx
from typing import Dict, List, Tuple
from dataset import Dataset, Outcome
import time

class RankSolver:
    def __init__(self, agents: List[int], categories: List[Dict]):
        """
        Rankメカニズムの初期化
        :param agents: エージェントIDのリスト
        :param categories: 各カテゴリの定員と優先順位
        """
        self.agents = agents
        self.categories = categories

    def solve(self, agent_ranks: Dict[int, List[int]]):
        """
        Rankメカニズムの実行
        :param agent_ranks: エージェントの希望順位リスト {agent_id: [cat_id1, cat_id2, ...]}
        """
        # 全体で一つの二部グラフを構築（最大マッチングを求めるため）
        G = nx.Graph()
        
        # エージェントノードの追加
        for a_id in self.agents:
            G.add_node(f"A_{a_id}", bipartite=0)
            
        # カテゴリスロットノードの追加
        for cat in self.categories:
            c_id = cat['id']
            for slot in range(cat['capacity']):
                slot_node = f"C_{c_id}__S_{slot}"
                G.add_node(slot_node, bipartite=1)
                
                # 適格性（Eligibility）と希望（Rank）に基づくエッジ
                # エージェントがそのカテゴリを「受け入れ可能（Rankリストに含まれる）」としている場合のみ接続
                for a_id in self.agents:
                    if a_id in cat['priority'] and c_id in agent_ranks.get(a_id, []):
                        # 重み付け（希望順位が高いほど重みを大きくする）
                        # Rank 0 (第1希望) -> 重み 100, Rank 1 -> 重み 99...
                        rank_idx = agent_ranks[a_id].index(c_id)
                        weight = 100 - rank_idx
                        G.add_edge(f"A_{a_id}", slot_node, weight=weight)

        # 最大重み最大マッチング（Max Weight Maximum Cardinality Matching）を計算
        # これにより「マッチング数を最大化」しつつ「より高いRank」が優先される
        matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)
        
        final_matching = {}
        for u, v in matching:
            if u.startswith("A_"):
                a_id = int(u.split("_")[1])
                c_id = int(v.split("__")[0].split("_")[1])
                final_matching[a_id] = c_id
            else:
                a_id = int(v.split("_")[1])
                c_id = int(u.split("__")[0].split("_")[1])
                final_matching[a_id] = c_id
                
        return final_matching
    
def execute_rank_on_dataset(dataset: Dataset) -> Outcome:
    agents_obj, categories_obj = dataset.to_algorithm_inputs()
    agent_ids = [ag.agent_id for ag in agents_obj.agents]
    
    # データセットの preference（選好）を Rank リストとして使用
    agent_ranks = {ag.agent_id: ag.preferences for ag in agents_obj.agents}
    
    cat_dicts = [{
        'id': c.category_id,
        'capacity': c.capacity,
        'priority': c.priority
    } for c in categories_obj]
    
    solver = RankSolver(agent_ids, cat_dicts)
    matching_dict = solver.solve(agent_ranks)
    
    return Outcome(dataset_id=dataset.id, algorithm_name="Rank", matching=matching_dict)

