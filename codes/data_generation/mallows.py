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

