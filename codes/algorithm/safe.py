import networkx as nx
from typing import Dict, List, Tuple
from codes.data_generation.dataset import Dataset
from codes.agent import Agents, Category, Outcome

class SafeSolver:
    def __init__(self, agents: List[int], categories: List[Dict]):
        """
        Safeメカニズムの初期化
        :param agents: エージェントIDのリスト
        :param categories: カテゴリ情報のリスト。各辞書は以下の形式:
               {
                 'id': 1, 
                 'type': 'regional', # または 'general'
                 'capacity': 5, 
                 'priority': [10, 3, 1, ...] # 優先順位リスト
               }
        """
        self.agents = agents
        self.categories = categories

    def _build_bipartite_graph(self, target_agents: List[int], target_categories: List[Dict]):
        """
        特定の枠（地域枠のみ、または一般枠のみ）に対する二部グラフを構築
        """
        G = nx.Graph()
        G.add_nodes_from([f"A_{a}" for a in target_agents], bipartite=0)
        
        category_slots = []
        for cat in target_categories:
            c_id = cat['id']
            # 定員分だけスロットを複製
            for slot in range(cat['capacity']):
                slot_node = f"C_{c_id}__S_{slot}"
                category_slots.append(slot_node)
                
                # 適格なエージェントとの間にエッジを張る
                # Safeメカニズムでは、そのカテゴリを「選択した」人だけが対象 [cite: 200]
                for agent_id in target_agents:
                    if agent_id in cat['priority']:
                        G.add_edge(f"A_{agent_id}", slot_node)
        
        G.add_nodes_from(category_slots, bipartite=1)
        return G

    def solve(self, agent_choices: Dict[int, int]):
        """
        Safeメカニズムの実行
        :param agent_choices: エージェントが選択したカテゴリIDの辞書 {agent_id: category_id}
        """
        final_matching = {}
        
        # 1. 各カテゴリごとに独立して最大マッチングを計算
        # Safeメカニズムの定義に基づき、カテゴリ間の干渉を防ぐ 
        for cat in self.categories:
            c_id = cat['id']
            
            # このカテゴリを選択したエージェントを抽出
            chosen_agents = [a for a, choice_id in agent_choices.items() if choice_id == c_id]
            
            if not chosen_agents:
                continue
                
            # カテゴリ単体での二部グラフ構築
            G = self._build_bipartite_graph(chosen_agents, [cat])
            
            # 最大マッチングを計算 (Hopcroft-Karp等)
            # 各カテゴリ内での最大サイズを保証する [cite: 150]
            raw_matching = nx.bipartite.maximum_matching(G)
            
            # 結果を整理
            for u, v in raw_matching.items():
                if u.startswith("A_"):
                    a_id = int(u.split("_")[1])
                    # カテゴリノード名 "C_1__S_0" からIDを抽出
                    c_id_matched = int(v.split("__")[0].split("_")[1])
                    final_matching[a_id] = c_id_matched
                    
        return final_matching

def execute_safe_mechanism(dataset_input, agent_choices: Dict[int, int]) -> List[Tuple[int, int]]:
    """
    既存のデータセット形式からSafeメカニズムを実行するラッパー
    """
    # 簡略化のため、dataset_inputから必要な形式に変換
    agents = [ag.agent_id for ag in dataset_input.agents]
    
    # カテゴリ情報の構築 (地域枠/一般枠の属性を付与)
    categories = []
    for cat in dataset_input.categories:
        categories.append({
            'id': cat.category_id,
            'capacity': cat.capacity,
            'priority': cat.priority,
            'type': getattr(cat, 'type', 'general') # デフォルトは一般
        })
        
    solver = SafeSolver(agents, categories)
    matching_dict = solver.solve(agent_choices)
    
    return list(matching_dict.items())

def execute_safe_on_dataset(dataset: Dataset, strategy="first_eligible") -> Outcome:
    """
    バッチフレームワーク用のSafeメカニズム実行関数
    """
    agents_obj, categories_obj = dataset.to_algorithm_inputs()
    agent_ids = [ag.agent_id for ag in agents_obj.agents]
    
    # Safeメカニズム特有の「選択（Choice）」ステップをシミュレート
    # 実験用として、各エージェントが「最初に見つけた適格な枠」を選択すると仮定
    agent_choices = {}
    for ag in agents_obj.agents:
        for cat in categories_obj:
            if ag.agent_id in cat.eligible_agents:
                agent_choices[ag.agent_id] = cat.category_id
                break
    
    # 前述のSafeSolverを呼び出し
    from codes.algorithm.safe import SafeSolver
    
    # 内部形式への変換
    cat_dicts = [{
        'id': c.category_id,
        'capacity': c.capacity,
        'priority': c.priority
    } for c in categories_obj]
    
    solver = SafeSolver(agent_ids, cat_dicts)
    matching_dict = solver.solve(agent_choices)
    
    # 他のアルゴリズムと同じOutcomeオブジェクトで返す
    from codes.agent import Outcome
    return Outcome(dataset_id=dataset.id, algorithm_name="Safe", matching=matching_dict)