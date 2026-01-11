import networkx as nx
from typing import List, Set, Dict

class SCUSolver:
    # 引数をテストコード(tests/test_scu.py:38)と同じに
    def __init__(self, agents, categories, capacities, priorities, precedence, beneficial):
        """
        Store SCU instance data: agents/categories, capacities, priorities, precedence,
        and the set of beneficial categories.
        """
        self.agent_ids = agents
        self.category_ids = categories
        self.capacities = capacities  # Dict[id, cap]
        self.priorities = priorities  # Dict[cat_id, List[ag_id]]
        self.precedence = precedence  # List[cat_id]
        self.beneficial_ids = set(beneficial)

    def solve(self) -> Dict[any, any]:
        """
        Compute the SCU matching by iterating categories in precedence order and
        assigning agents when feasibility can be preserved.
        """
        # ステップ1 & 2: m (最大基数) と b (最大受益者数) を計算 [cite: 531-534]
        b = self._compute_max_beneficial()
        m = self._compute_max_cardinality(b)
        
        X = {} # 確定したマッチング {agent_id: category_id}
        
        # ステップ3: カテゴリの処理順に従ってループ [cite: 368, 535]
        for cat_id in self.precedence:
            # カテゴリ内のエージェントを優先順位順にチェック [cite: 369]
            if cat_id not in self.priorities: continue
            
            for ag_id in self.priorities[cat_id]:
                if ag_id in X: continue
                
                # 割り当て可能か判定 [cite: 370-374, 546-548]
                if self._can_assign(ag_id, cat_id, X, b, m):
                    X[ag_id] = cat_id
                    # カテゴリ定員チェック
                    if list(X.values()).count(cat_id) >= self.capacities[cat_id]:
                        break
        return X

    def _can_assign(self, current_ag, current_cat, X, b, m):
        """
        Check whether assigning (current_ag -> current_cat) can still achieve
        total matching size m with b beneficial matches via a flow test.
        """
        G = nx.DiGraph()
        S, T = "source", "sink"
        C_beneficial, C_open = "C_star", "C_zero"
        
        # ターゲットノードの設定 [cite: 526-529]
        G.add_edge(C_beneficial, T, capacity=b)
        G.add_edge(C_open, T, capacity=(m - b))
        
        test_assignments = {**X, current_ag: current_cat}
        
        for ag_id in self.agent_ids:
            G.add_edge(S, ag_id, capacity=1)
            if ag_id in test_assignments:
                G.add_edge(ag_id, test_assignments[ag_id], capacity=1)
            else:
                for c_id in self.category_ids:
                    # エージェントがそのカテゴリに対して適格(priorityに含まれる)か確認 [cite: 88, 524]
                    if ag_id in self.priorities.get(c_id, []):
                        G.add_edge(ag_id, c_id, capacity=1)
        
        for c_id in self.category_ids:
            target = C_beneficial if c_id in self.beneficial_ids else C_open
            G.add_edge(c_id, target, capacity=self.capacities[c_id])
            
        try:
            flow_val, _ = nx.maximum_flow(G, S, T)
            return flow_val == m
        except:
            return False

    def _compute_max_beneficial(self):
        """
        Compute the maximum number of matches that can be assigned to
        beneficial categories.
        """
        G = nx.DiGraph()
        S, T = "source", "sink"
        for ag_id in self.agent_ids:
            G.add_edge(S, ag_id, capacity=1)
            for c_id in self.beneficial_ids:
                if ag_id in self.priorities.get(c_id, []):
                    G.add_edge(ag_id, c_id, capacity=1)
        for c_id in self.beneficial_ids:
            G.add_edge(c_id, T, capacity=self.capacities[c_id])
        val, _ = nx.maximum_flow(G, S, T)
        return val

    def _compute_max_cardinality(self, b):
        """
        Compute the maximum total matching size (cardinality).
        """
        G = nx.DiGraph()
        S, T = "source", "sink"
        for ag_id in self.agent_ids:
            G.add_edge(S, ag_id, capacity=1)
            for c_id in self.category_ids:
                if ag_id in self.priorities.get(c_id, []):
                    G.add_edge(ag_id, c_id, capacity=1)
        for c_id in self.category_ids:
            G.add_edge(c_id, T, capacity=self.capacities[c_id])
        val, _ = nx.maximum_flow(G, S, T)
        return val
