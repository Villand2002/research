import pandas as pd
import numpy as np
from typing import List, Tuple
from scipy.stats import kendalltau
from typing import List, Dict, Tuple
import networkx as nx


class Agent:
    def __init__(self, agent_id: int, acceptable_categories: List[int]):
        self.agent_id = agent_id
        self.acceptable_categories = acceptable_categories

    def show_info(self):
        print(f"Agent ID: {self.agent_id}, Acceptable Categories: {self.acceptable_categories}")
        
class Agents:
    def __init__(self, agents: List[Agent], agent_number: int):
        self.agents = agents
        self.agent_number = agent_number

    def show_all_info(self):
        for agent in self.agents:
            agent.show_info()

class Category:
    def __init__(self, category_id: int, capacity: int, priority: List[int]):
        self.category_id = category_id
        self.capacity = capacity
        self.priority = priority

    def show_info(self):
        print(f"Category ID: {self.category_id}, Capacity: {self.capacity}, Priority: {self.priority}")


def kendall_distance(perm1: List[int], perm2: List[int]) -> int:
    """2つの順列間のKendall距離を計算"""
    n = len(perm1)
    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            # perm1での順序とperm2での順序が逆転している場合カウント
            idx1_i = perm1.index(perm2[i])
            idx1_j = perm1.index(perm2[j])
            if idx1_i > idx1_j:
                distance += 1
    return distance

def generate_mallows_permutation(reference: List[int], phi: float) -> List[int]:
    """
    Mallows modelに基づいて順列を生成
    
    Args:
        reference: 基準となる順列
        phi: 分散パラメータ (0 < phi <= 1)
             phi=1: 基準順列と同じ, phi→0: ランダムに近づく
    
    Returns:
        生成された順列
    """
    n = len(reference)
    result = []
    remaining = reference.copy()
    
    for i in range(n):
        m = len(remaining)
        # 各位置の確率を計算
        probs = []
        for j in range(m):
            # 位置jに挿入した場合のスコア
            score = phi ** j
            probs.append(score)
        
        # 正規化
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        # 確率的に位置を選択
        pos = np.random.choice(m, p=probs)
        result.append(remaining[pos])
        remaining.pop(pos)
    
    return result

def generate_priorities(agent_number: int, category_number: int, 
                       phi: float = 0.5, reference: List[int] = None) -> pd.DataFrame:
    """
    Mallows modelを使ってエージェントの選好を生成
    
    Args:
        agent_number: エージェント数
        category_number: カテゴリ数
        phi: Mallowsモデルの分散パラメータ (0 < phi <= 1)
        reference: 基準となる順列。Noneの場合はランダムに生成
    
    Returns:
        エージェントの選好を表すDataFrame (行: エージェント, 列: 順位)
    """
    # 基準順列を設定
    if reference is None:
        reference = list(range(category_number))
    
    # 各エージェントの選好を生成
    preferences = []
    for i in range(agent_number):
        pref = generate_mallows_permutation(reference, phi)
        preferences.append(pref)
    
    # DataFrameに変換
    df = pd.DataFrame(preferences, 
                     columns=[f'rank_{i+1}' for i in range(category_number)],
                     index=[f'agent_{i+1}' for i in range(agent_number)])
    
    return df

def generate_capacities(category_number: int, total_capacity: int = None,
                       distribution: str = 'uniform') -> pd.DataFrame:
    """
    カテゴリの受け入れ人数を生成
    
    Args:
        category_number: カテゴリ数
        total_capacity: 総受け入れ人数。Noneの場合は自動設定
        distribution: 分布タイプ ('uniform', 'normal', 'exponential')
    
    Returns:
        各カテゴリの受け入れ人数を表すDataFrame
    """
    if total_capacity is None:
        total_capacity = category_number * 10  # デフォルト値
    
    if distribution == 'uniform':
        # 均等分配
        base = total_capacity // category_number
        remainder = total_capacity % category_number
        capacities = [base] * category_number
        for i in range(remainder):
            capacities[i] += 1
    
    elif distribution == 'normal':
        # 正規分布に基づく分配
        raw = np.random.normal(loc=total_capacity/category_number, 
                              scale=total_capacity/(category_number*3), 
                              size=category_number)
        raw = np.maximum(raw, 1)  # 最小値を1に
        capacities = (raw / raw.sum() * total_capacity).astype(int)
        # 合計を調整
        diff = total_capacity - capacities.sum()
        capacities[0] += diff
    
    elif distribution == 'exponential':
        # 指数分布（人気度の差が大きい場合）
        raw = np.random.exponential(scale=1, size=category_number)
        raw = np.sort(raw)[::-1]  # 降順
        capacities = (raw / raw.sum() * total_capacity).astype(int)
        # 合計を調整
        diff = total_capacity - capacities.sum()
        capacities[0] += diff
    
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    df = pd.DataFrame({
        'category': [f'category_{i}' for i in range(category_number)],
        'capacity': capacities
    })
    
    return df

def analyze_preferences(preferences_df: pd.DataFrame) -> dict:
    """選好の統計情報を計算"""
    n_agents, n_categories = preferences_df.shape
    
    # 各カテゴリが各順位に現れる頻度
    rank_distribution = {}
    for rank_col in preferences_df.columns:
        rank_distribution[rank_col] = preferences_df[rank_col].value_counts().to_dict()
    
    # 平均Kendall距離（最初のエージェントを基準）
    reference = preferences_df.iloc[0].tolist()
    distances = []
    for i in range(1, n_agents):
        pref = preferences_df.iloc[i].tolist()
        dist = kendall_distance(reference, pref)
        distances.append(dist)
    
    return {
        'n_agents': n_agents,
        'n_categories': n_categories,
        'avg_kendall_distance': np.mean(distances) if distances else 0,
        'std_kendall_distance': np.std(distances) if distances else 0,
        'rank_distribution': rank_distribution
    }



# REVアルゴリズム

import networkx as nx

def assign_eligibility_random(agents_obj: Agents, categories_obj: List[Category], p=0.4):
    """
    Agents と Category の構造に合わせて eligibility を設定
    """
    for cat in categories_obj:
        eligible = []
        for ag in agents_obj.agents:
            if np.random.rand() < p:
                eligible.append(ag.agent_id)
        if len(eligible) == 0:
            eligible.append(np.random.choice([ag.agent_id for ag in agents_obj.agents]))
        # Category 内に保持
        cat.eligible_agents = eligible  # ← 新しいフィールドを追加

def assign_category_priorities(agents_obj: Agents, categories_obj: List[Category], phi=0.6):
    """
    各カテゴリに対して agent の優先順位を Mallows で生成
    """
    reference = [ag.agent_id for ag in agents_obj.agents]

    for cat in categories_obj:
        perm = generate_mallows_permutation(reference, phi)
        cat.priority = perm  # 上 → 下 = 高優先 → 低優先


def generate_category_priorities(agents: List[str], categories: List[str], phi: float = 0.5, reference: List[str] = None):
    """
    各カテゴリの優先順位 (category → agent の順番) を Mallows model で生成する
    
    Args:
        agents: エージェントのリスト
        categories: カテゴリのリスト
        phi: Mallowsの集中度 (1.0に近いほど reference に近い)
        reference: 基準となる agent 順列（None なら agent 順番そのまま）
    
    Returns:
        priority: Dict[category] = List[agent (優先順位高い→低い)]
    """
    if reference is None:
        reference = agents.copy()  # baseline 順序をそのまま基準とする
    
    priority = {}
    for c in categories:
        perm = generate_mallows_permutation(reference, phi)
        priority[c] = perm  # 上にあるほど優先度が高い
    
    return priority


def compute_max_matching_size(agents_obj: Agents, categories_obj: List[Category], banned_agents: set):
    """
    クラス構造に合わせた最大マッチング計算
    """
    G = nx.DiGraph()
    S, T = "S", "T"
    G.add_node(S)
    G.add_node(T)

    # agent ノード
    for ag in agents_obj.agents:
        if ag.agent_id not in banned_agents:
            G.add_edge(S, ag.agent_id, capacity=1)

    # category ノード
    for cat in categories_obj:
        G.add_edge(cat.category_id, T, capacity=cat.capacity)

    # エッジ追加 (eligibility + priority-based envy rule)
    for cat in categories_obj:
        rank = {a: i for i, a in enumerate(cat.priority)}  # priority index
        for ag in cat.eligible_agents:  # eligible list from assign step
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


def rev_algorithm(agents_obj: Agents, categories_obj: List[Category]):
    agents_list = [ag.agent_id for ag in agents_obj.agents]
    banned = set()

    base_val = compute_max_matching_size(agents_obj, categories_obj, banned)

    # baseline は agent_id の昇順を弱者側とみなす
    for ag in reversed(agents_list):
        test = banned | {ag}
        val = compute_max_matching_size(agents_obj, categories_obj, test)
        if val == base_val:
            banned = test

    # 最終マッチング構築
    # （同じグラフ作りなおしてフロー辞書から抽出）
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



def nx_rebuild_graph(agents_obj: Agents, categories_obj: List[Category], banned_agents):
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


# MMA
def execute_mma(agents_obj: Agents, categories_obj: List[Category]) -> List[Tuple[int, int]]:
    """
    Maximum Matching Adjustment (MMA) Algorithm
    """
    agents = agents_obj.agents
    agent_ids = [ag.agent_id for ag in agents]

    # Eligibility Graph G =を構築
    G = nx.Graph()
    for ag in agents:
        for cat in categories_obj:
            if ag.agent_id in cat.eligible_agents:     # eligibility edge
                G.add_edge(f"A_{ag.agent_id}", f"C_{cat.category_id}")

    #Step 2: 最大マッチング μ を求める 
    max_matching = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    # μ の内部表現 {agent → category}
    mu: Dict[int, int] = {}

    for a, c in max_matching:
        if a.startswith("A_"):
            agent = int(a[2:])
            category = int(c[2:])
        else:
            agent = int(c[2:])
            category = int(a[2:])
        mu[agent] = category

    # 現在 unmatched の agent をリストで持つ
    unmatched_agents = [ag for ag in agent_ids if ag not in mu]

    for i in unmatched_agents:
        # i が eligible なカテゴリを取得（順番が必要）
        cats = [cat for cat in categories_obj if i in cat.eligible_agents]

        for c in cats:
            cat_id = c.category_id

            # category cat_id に現在入っている agent を列挙
            matched_agents = [a for a, cat in mu.items() if cat == cat_id]

            if len(matched_agents) < c.capacity:
                # まだ空きがある → i を追加
                mu[i] = cat_id
                break

            # 削除されるのは最も priority の低い agent i'
            # priority はリストの後ろほど priority が低いとする
            priorities = c.priority
            i_prime = max(matched_agents, key=lambda a: priorities.index(a))

            # i の方が優先度高い？
            if priorities.index(i) < priorities.index(i_prime):
                # 追い出し処理
                mu[i] = cat_id
                del mu[i_prime]
                break

    # --- 最終的なマッチングを (agent, category) のリストで返す ---
    return list(mu.items())



# 使用例
if __name__ == "__main__":
    # パラメータ設定
    n_agents = 10
    n_categories = 5
    phi = 0.7  # 高いほど基準順列に近い
    
    # 選好の生成
    print("=" * 50)
    print("エージェントの選好生成 (Mallows Model)")
    print("=" * 50)
    preferences = generate_priorities(n_agents, n_categories, phi=phi)
    print(f"\n最初の10エージェントの選好:\n{preferences.head(10)}")
    
    # 統計情報
    stats = analyze_preferences(preferences)
    print(f"\n統計情報:")
    print(f"- エージェント数: {stats['n_agents']}")
    print(f"- カテゴリ数: {stats['n_categories']}")
    print(f"- 平均Kendall距離: {stats['avg_kendall_distance']:.2f}")
    print(f"- Kendall距離の標準偏差: {stats['std_kendall_distance']:.2f}")
    
    # カテゴリ受け入れ人数の生成
    print("\n" + "=" * 50)
    print("カテゴリの受け入れ人数生成")
    print("=" * 50)
    
    total_capacity = 80  # 総受け入れ人数
    
    print("\n1. 均等分配:")
    capacities_uniform = generate_capacities(n_categories, total_capacity, 'uniform')
    print(capacities_uniform)
    
    print("\n2. 正規分布:")
    capacities_normal = generate_capacities(n_categories, total_capacity, 'normal')
    print(capacities_normal)
    
    print("\n3. 指数分布 (人気度差大):")
    capacities_exp = generate_capacities(n_categories, total_capacity, 'exponential')
    print(capacities_exp)
    
    # CSVに保存
    preferences.to_csv('preferences.csv')
    capacities_uniform.to_csv('capacities.csv', index=False)
    print("\n結果をpreferences.csvとcapacities.csvに保存しました")
    
    # REVアルゴリズムの実行例
   
    # capacities_uniform の index がカテゴリ番号として使える
    cap_dict = {i: capacities_uniform.loc[i, "capacity"] for i in range(n_categories)}

    categories_obj = [
        Category(category_id=i, capacity=cap_dict[i], priority=[])
        for i in range(n_categories)
    ]

    # Agents オブジェクト生成
    agents_obj = Agents(
        agents=[Agent(agent_id=i, acceptable_categories=[]) for i in range(1, n_agents+1)],
        agent_number=n_agents
    )


    # eligibility をカテゴリ側に付与
    assign_eligibility_random(agents_obj, categories_obj, p=0.3)

    # Mallows による category → agent の優先順位生成
    assign_category_priorities(agents_obj, categories_obj, phi=0.6)

    # --- REV を実行 --- #
    matching, rejected = rev_algorithm(agents_obj, categories_obj)

    print("Rejected agents:", rejected)
    print("\nMatching result:")
    for a, c in matching:
        print(f"Agent {a} → Category {c}")
        
    # MMAアルゴリズムの実行例
    mma_matching = execute_mma(agents_obj, categories_obj)
    print("\nMMA Matching result:")
    for a, c in mma_matching:
        print(f"Agent {a} → Category {c}")
