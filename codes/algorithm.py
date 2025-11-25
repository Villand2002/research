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

import networkx as nx

class MMASolver:
    def __init__(self, agents, categories, capacities, priorities):
        """
        初期化
        :param agents: エージェントのリスト (例: ['i1', 'i2', ...])
        :param categories: カテゴリのリスト (例: ['c1', 'c2', ...])
        :param capacities: 各カテゴリの定員 (例: {'c1': 1, 'c2': 2})
        :param priorities: 各カテゴリの優先順位リスト (例: {'c1': ['i2', 'i1', ...]})
                           リストの先頭ほど優先順位が高いとする
        """
        self.agents = agents
        self.categories = categories
        self.capacities = capacities
        self.priorities = priorities

    def _build_eligibility_graph(self):
        """
        適格性グラフの構築 (Step 1用)
        NetworkXの最大マッチングは1対1なので、定員q > 1の場合、
        カテゴリノードを複製(c_0, c_1...)して対応する。
        """
        G = nx.Graph()
        G.add_nodes_from(self.agents, bipartite=0)
        
        category_nodes = []
        for c in self.categories:
            # 定員分だけノードを複製
            for slot in range(self.capacities.get(c, 0)):
                node_name = f"{c}__slot__{slot}"
                category_nodes.append(node_name)
                
                # 適格なエージェント(優先順位リストにいる人)との間にエッジを張る 
                for agent in self.priorities.get(c, []):
                    if agent in self.agents:
                        G.add_edge(agent, node_name)
        
        G.add_nodes_from(category_nodes, bipartite=1)
        return G

    def _get_initial_max_matching(self):
        """
        Step 1: 最大サイズのマッチングを計算 
        """
        G = self._build_eligibility_graph()
        # Hopcroft-Karp algorithm等で最大マッチングを計算
        raw_matching = nx.bipartite.maximum_matching(G, top_nodes=self.agents)
        
        # マッチング形式を {Agent: Category} に整理
        matching = {}
        for u, v in raw_matching.items():
            if u in self.agents:
                # v は "c1__slot__0" のような形式なので "c1" に戻す
                cat = v.split("__slot__")[0]
                matching[u] = cat
        return matching

    def solve(self):
        """
        MMAルールの実行 (Algorithm 3)
        """
        # Step 1: 最大マッチングの計算
        mu = self._get_initial_max_matching()
        
        # Step 2: Priorityに基づく調整 (Adjustment) [cite: 191]
        while True:
            changed = False
            # 現在マッチしていないエージェントを取得
            unmatched_agents = [a for a in self.agents if a not in mu]
            
            for i in unmatched_agents:
                # エージェントiが適格なカテゴリを探索
                # (優先順位リストに含まれていれば適格 [cite: 88])
                eligible_cats = [c for c in self.categories if i in self.priorities.get(c, [])]
                
                for c in eligible_cats:
                    # カテゴリcに現在マッチしているエージェントたちを取得
                    current_assigned = [a for a, cat in mu.items() if cat == c]
                    
                    if not current_assigned:
                        continue

                    # カテゴリcの中で「最も優先順位が低い」エージェント(i_prime)を探す 
                    # prioritiesリストのindexが大きいほど優先順位が低い
                    c_priority_list = self.priorities[c]
                    
                    # 現在マッチしている人の中で、優先順位リスト上のインデックスが最大の人が「最弱」
                    # (リストにいない人は除外するが、適格性グラフの定義上全員いるはず)
                    i_prime = max(current_assigned, key=lambda x: c_priority_list.index(x))
                    
                    # 比較: i が i_prime より優先順位が高いか？
                    # (indexが小さい方が優先順位が高い)
                    idx_i = c_priority_list.index(i)
                    idx_prime = c_priority_list.index(i_prime)
                    
                    if idx_i < idx_prime: # i has higher priority 
                        # Swap実行
                        print(f"Swap: {i} replaces {i_prime} in {c}")
                        mu[i] = c           # i をマッチさせる
                        del mu[i_prime]     # i' をマッチングから削除（Unmatchedへ）
                        changed = True
                        break # マッチングが変わったので、whileループの先頭(unmatchedの再評価)へ戻るのが安全
                
                if changed:
                    break
            
            if not changed:
                break # 変更がなくなれば終了
                
        return mu



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
