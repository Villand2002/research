from typing import List

import numpy as np


# --- 改善された関数 ---
def generate_mallows_permutation(reference: List[int], phi: float) -> List[int]:
    """Generate a Mallows permutation via the repeated insertion method (RIM)."""

    m = len(reference)
    ranking: List[int] = []
    for i in range(m):
        size = i + 1
        # phi==1 の場合でも浮動小数で計算することで正規化を安定化させる
        probs = np.array([phi ** (size - j - 1) for j in range(size)], dtype=float)
        probs /= probs.sum()
        idx = np.random.choice(size, p=probs)
        ranking.insert(idx, reference[i])

    return ranking

