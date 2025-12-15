import networkx as nx
from codes.agent import Agents, Category


class SCUSolver:
    def __init__(self, agents, categories, capacities, priorities, precedence_order, beneficial_categories):
        """
        SCU (Sequential Category Updating) Rule Solver
        
        Parameters:
        - agents: List of agent IDs
        - categories: List of category IDs
        - capacities: Dict {category: capacity}
        - priorities: Dict {category: [agent_list_sorted_by_priority]}
        - precedence_order: List of categories in processing order (earlier is higher precedence)
        - beneficial_categories: Set or List of categories considered "beneficial" (C*)
        """
        self.agents = agents
        self.categories = categories
        self.capacities = capacities
        self.priorities = priorities
        self.precedence = precedence_order
        self.beneficial_cats = set(beneficial_categories)
        self.open_cats = set(categories) - self.beneficial_cats

    def _build_base_graph(self, current_assignments=None):
        """
        フローネットワークの基本構造を作成
        current_assignments: 既に確定した {agent: category} の辞書
        """
        if current_assignments is None:
            current_assignments = {}
            
        assigned_agents = set(current_assignments.keys())
        
        # グラフ構築
        G = nx.DiGraph()
        S, T = 'source', 'sink'
        C_0_NODE, C_STAR_NODE = 'C_0_super', 'C_star_super'
        
        # 1. Source -> Agents (未割り当てのエージェントのみ)
        # 既に割り当てられたエージェントは、後のロジックで「確定枠」として処理するためグラフには含めない
        # （あるいは含めて強制的に流す方法もあるが、残余グラフで考える方がシンプル）
        remaining_agents = [a for a in self.agents if a not in assigned_agents]
        for agent in remaining_agents:
            G.add_edge(S, agent, capacity=1)
            
        # 2. Agents -> Categories
        for cat in self.categories:
            # カテゴリの現在の残り容量を計算
            # (基本容量) - (既にこのカテゴリに確定した人数)
            assigned_count = sum(1 for c in current_assignments.values() if c == cat)
            current_cap = self.capacities.get(cat, 0) - assigned_count
            
            # 容量が残っている場合のみエッジを張る
            # ただし、MaxFlowの計算上、カテゴリノード自体の容量制限が必要
            # NetworkXではノード容量を直接扱えないため、
            # Agent -> CatNode -> SuperNode の SuperNodeへのエッジで容量を制限する
            
            # Agent -> Category エッジ
            # 優先順位リストに載っている(適格な)エージェントのみ
            for agent in self.priorities.get(cat, []):
                if agent in remaining_agents:
                    G.add_edge(agent, cat, capacity=1)

            # 3. Categories -> Super Nodes (C_0 or C*)
            # ここでカテゴリごとの容量制限をかける
            if cat in self.beneficial_cats:
                G.add_edge(cat, C_STAR_NODE, capacity=current_cap)
            else:
                G.add_edge(cat, C_0_NODE, capacity=current_cap)
                
        return G, S, T, C_0_NODE, C_STAR_NODE

    def _calc_max_values(self):
        """
        Step 1 & 2: 
        b = 最大受益者割当数 (Max flow to C*)
        m = 最大全体割当数 (Max total flow given b)
        """
        G, S, T, C0, CS = self._build_base_graph()
        
        # --- Step 1: Compute b (Max flow to C*) ---
        # C0 -> T の容量を 0 にして、C* へのフローを最大化
        G.add_edge(C0, T, capacity=0)
        G.add_edge(CS, T, capacity=float('inf'))
        
        b = nx.maximum_flow(G, S, T)[0]
        
        # --- Step 2: Compute m (Max total flow given b) ---
        # 論文では "Set lower bound of (C*, T) to b" とあるが、
        # MaxFlowアルゴリズムの性質上、まずC*へのフローを確保しつつ全体を最大化するのは
        # 適切な容量設定でシミュレートできる。
        # ここでは単純に全体の最大フローを計算すれば、Lemma 1により整合性が保証される。
        # (ただし、bが確保できることは確認済み)
        
        # 容量制限を解除/設定
        # C0 -> T: 無限 (or 全エージェント数)
        # C* -> T: 無限 (or 全エージェント数)
        # 単純な最大マッチング数を求めればよい
        G.remove_edge(C0, T)
        G.remove_edge(CS, T)
        G.add_edge(C0, T, capacity=float('inf'))
        G.add_edge(CS, T, capacity=float('inf'))
        
        m = nx.maximum_flow(G, S, T)[0]
        
        return int(b), int(m)

    def _can_assign(self, agent, category, current_assignments, target_b, target_total):
        """
        Step 3の判定ロジック:
        agent を category に割り当てた状態で、
        残りのエージェントを使って「目標とするBeneficial数(target_b)」と「目標とする全体数(target_total)」
        を達成できるか判定する。
        """
        # 仮の割り当てを作成
        temp_assignments = current_assignments.copy()
        temp_assignments[agent] = category
        
        # 残りの必要数を計算
        # 既に割り当てられた人たち + 今回の仮割り当て
        current_b_count = sum(1 for c in temp_assignments.values() if c in self.beneficial_cats)
        current_total_count = len(temp_assignments)
        
        rem_b_needed = target_b - current_b_count
        rem_total_needed = target_total - current_total_count
        
        # 目標を超えてしまっていたら不可能 (例: C*枠がもう一杯なのにC*に入れようとした)
        # ただし b は「最大」なので、それ以下になることは許容されない（Max Cardinality維持のため）
        # 論文のアルゴリズムは「Lower bound」を設定するので、ピッタリ達成できるかを見る
        if rem_b_needed < 0: return False # あり得ないが念のため
        # C0への必要数 (全体 - C*)
        rem_open_needed = rem_total_needed - rem_b_needed
        if rem_open_needed < 0: return False

        # 残余グラフを構築してチェック
        G, S, T, C0, CS = self._build_base_graph(temp_assignments)
        
        # Sinkへのエッジ容量を「残りの必要数」に制限する
        # これでMaxFlowがこの容量一杯まで流れれば、目標達成可能ということ
        G.add_edge(C0, T, capacity=rem_open_needed)
        G.add_edge(CS, T, capacity=rem_b_needed)
        
        flow_val = nx.maximum_flow(G, S, T)[0]
        
        # フローが目標値 (残りの必要総数) に達しているか？
        return flow_val == rem_total_needed

    def solve(self):
        """
        Algorithm 5: Main Procedure
        """
        # 1. b (最大受益者数) と m (最大全体数) を計算
        b, m = self._calc_max_values()
        print(f"Calculated Max Values -> m (Total): {m}, b (Beneficial): {b}")
        
        # 2. Sequential Updating Procedure
        # X: 確定した割り当て {agent: category}
        X = {}
        
        # カテゴリをPrecedence順に処理
        for c in self.precedence:
            # このカテゴリに適格なエージェントを優先順位順に取得
            # 既にXに含まれているエージェントは除外
            eligible_agents = [a for a in self.priorities.get(c, []) if a not in X]
            
            # 各エージェントについて判定
            for i in eligible_agents:
                # カテゴリの定員チェック (既に埋まっていたらスキップ)
                assigned_in_c = sum(1 for cat in X.values() if cat == c)
                if assigned_in_c >= self.capacities.get(c, 0):
                    break # このカテゴリは満員
                
                # 判定: i を c に割り当てても、最終的に m と b を達成できるか？
                if self._can_assign(i, c, X, b, m):
                    X[i] = c
                    # print(f"Assigned {i} -> {c}")
        
        return X

# ==========================================
# 実行例 (論文の Example 4 を再現)
# ==========================================
if __name__ == "__main__":
    # Example 4 Settings
    agents = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']
    categories = ['c1', 'c2', 'c3']
    
    # Beneficial: c1, c3
    # Open: c2
    beneficial = ['c1', 'c3']
    
    # Precedence: c1 > c2 > c3
    precedence = ['c1', 'c2', 'c3']
    
    capacities = {'c1': 1, 'c2': 1, 'c3': 1}
    
    # Priorities (論文 p.18)
    # c1: i2 > i1 > i3
    # c2: i2 > i4 > i6 > i3 > i5 > i1
    # c3: i5 > i4 > i6
    priorities = {
        'c1': ['i2', 'i1', 'i3'],
        'c2': ['i2', 'i4', 'i6', 'i3', 'i5', 'i1'],
        'c3': ['i5', 'i4', 'i6']
    }

    print("--- SCU Algorithm (Example 4) ---")
    scu = SCUSolver(agents, categories, capacities, priorities, precedence, beneficial)
    result = scu.solve()
    
    # 結果表示
    print("\nFinal Matching:")
    for a in sorted(result.keys()):
        print(f"{a} -> {result[a]}")