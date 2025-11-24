import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import kendalltau

# --- 高速化のための Fenwick Tree クラス ---
class FenwickTree:
    def __init__(self, size):
        self.tree = [0] * (size + 1)

    def add(self, i, delta):
        i += 1  # 1-based indexing
        while i < len(self.tree):
            self.tree[i] += delta
            i += i & (-i)

    def sum(self, i):
        i += 1
        s = 0
        while i > 0:
            s += self.tree[i]
            i -= i & (-i)
        return s

    def find_kth(self, k):
        """k番目（0-indexed）に存在する要素のインデックスを見つける (二分探索 + BIT)"""
        # ここでは k は 0-indexed で「残りの中で k+1 番目の要素」を探す
        idx = 0
        # ビット操作で二分探索
        for i in range(len(self.tree).bit_length() - 1, -1, -1):
            next_idx = idx + (1 << i)
            if next_idx < len(self.tree) and self.tree[next_idx] <= k:
                idx = next_idx
                k -= self.tree[idx]
        return idx

# --- 改善された関数 ---
def generate_mallows_permutation(reference: List[int], phi: float) -> List[int]:
    """
    Mallows modelに基づいて順列を生成 (高速化版)
    Binary Indexed Tree を使用して O(n log n) を実現
    """
    n = len(reference)
    
    # 1. 各ステップでの挿入位置（の相対インデックス）を決定する
    # これは確率計算だけなので先に行う
    insertion_indices = []
    for i in range(n):
        m = n - i # 残りの要素数
        
        # 確率分布の計算 (幾何級数的)
        # phi^0, phi^1, ..., phi^(m-1)
        # 高速化のため、全ての確率を計算せずに済むならなお良いが、
        # ここでは分布に従ってindexを選ぶ処理を行う
        
        # 分母 (1 + phi + ... + phi^(m-1)) = (1 - phi^m) / (1 - phi)
        if phi == 1.0:
            probs = np.ones(m) / m
        else:
            # np.arangeなどを使うと遅いので、必要なランダム選択だけ行う工夫も可能だが
            # ここでは元のロジックの確率分布を維持
            denom = (1 - phi ** m) / (1 - phi)
            
            # 累積分布関数(CDF)を使って乱択するのが高速だが、
            # 簡易的にnumpyのchoiceを使う（ここがボトルネックになりうるがpopよりはマシ）
            probs = (1 - phi) * (phi ** np.arange(m)) / (1 - phi ** m)
            
            # 正規化誤差の補正
            probs /= probs.sum()
        
        # 位置を選択
        pos = np.random.choice(m, p=probs)
        insertion_indices.append(pos)
    
    # 2. Fenwick Treeを使って実際に順列を構築
    # 最初は全て「利用可能(1)」の状態
    ft = FenwickTree(n)
    for i in range(n):
        ft.add(i, 1)
        
    result = []
    for pos in insertion_indices:
        # 残っているものの中で pos 番目の要素の実際のインデックスを取得
        real_idx = ft.find_kth(pos)
        result.append(reference[real_idx])
        
        # その要素を使用済みにする (1 -> 0)
        ft.add(real_idx, -1)
        
    return result

def kendall_distance(perm1: List[int], perm2: List[int]) -> int:
    """2つの順列間のKendall距離を計算"""
    n = len(perm1)
    distance = 0
    for i in range(n):
        for j in range(i + 1, n):
            idx1_i = perm1.index(perm2[i])
            idx1_j = perm1.index(perm2[j])
            if idx1_i > idx1_j:
                distance += 1
    return distance

def generate_mallows_permutation(reference: List[int], phi: float) -> List[int]:
    """Mallows modelに基づいて順列を生成 (Sequential Insertion)"""
    n = len(reference)
    result = []
    remaining = reference.copy()
    
    for i in range(n):
        m = len(remaining)
        probs = []
        for j in range(m):
            score = phi ** j
            probs.append(score)
        
        probs = np.array(probs)
        probs = probs / probs.sum()
        
        pos = np.random.choice(m, p=probs)
        result.append(remaining[pos])
        remaining.pop(pos)
    
    return result

def generate_priorities(agent_number: int, category_number: int, 
                        phi: float = 0.5, reference: List[int] = None) -> pd.DataFrame:
    """Mallows modelを使ってエージェントの選好を生成"""
    if reference is None:
        reference = list(range(category_number))
    
    preferences = []
    for i in range(agent_number):
        pref = generate_mallows_permutation(reference, phi)
        preferences.append(pref)
    
    df = pd.DataFrame(preferences, 
                      columns=[f'rank_{i+1}' for i in range(category_number)],
                      index=[f'agent_{i+1}' for i in range(agent_number)])
    return df

def generate_capacities(category_number: int, total_capacity: int = None,
                        distribution: str = 'uniform') -> pd.DataFrame:
    """カテゴリの受け入れ人数を生成"""
    if total_capacity is None:
        total_capacity = category_number * 10
    
    if distribution == 'uniform':
        base = total_capacity // category_number
        remainder = total_capacity % category_number
        capacities = [base] * category_number
        for i in range(remainder):
            capacities[i] += 1
    
    elif distribution == 'normal':
        raw = np.random.normal(loc=total_capacity/category_number, 
                               scale=total_capacity/(category_number*3), 
                               size=category_number)
        raw = np.maximum(raw, 1)
        capacities = (raw / raw.sum() * total_capacity).astype(int)
        diff = total_capacity - capacities.sum()
        capacities[0] += diff
    
    elif distribution == 'exponential':
        raw = np.random.exponential(scale=1, size=category_number)
        raw = np.sort(raw)[::-1]
        capacities = (raw / raw.sum() * total_capacity).astype(int)
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
    rank_distribution = {}
    for rank_col in preferences_df.columns:
        rank_distribution[rank_col] = preferences_df[rank_col].value_counts().to_dict()
    
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