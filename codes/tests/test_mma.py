import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.algorithm.mma import MMASolver

class TestMMASolver:
    def test_example1_paper(self):
        """
        Paper Example 1 & 2 [cite: 94-102, 196-198]
        MMAが「優先順位の尊重 (Respect for Priorities)」を満たすかテスト。
        初期マッチングがどうあれ、i2 (両方で最優先) は必ずマッチし、
        かつ優先順位の低いエージェントが不当に枠を占有しないことを確認。
        """
        agents = ['i1', 'i2', 'i3']
        categories = ['c1', 'c2']
        capacities = {'c1': 1, 'c2': 1}
        
        # 優先順位 (リストの先頭が高い)
        # c1: i2 > i1 > i3
        # c2: i2 > i3 > i1
        priorities = {
            'c1': ['i2', 'i1', 'i3'],
            'c2': ['i2', 'i3', 'i1']
        }
        
        solver = MMASolver(agents, categories, capacities, priorities)
        matching = solver.solve()
        
        # 検証 1: Maximum Cardinality [cite: 125]
        # 定員2, エージェント3名, 適格者十分 -> 必ず2名マッチする
        assert len(matching) == 2
        
        # 検証 2: Respect for Priorities [cite: 116]
        # i2はc1, c2の両方でトップ優先順位なので、必ずマッチしているはず
        assert 'i2' in matching
        
        # 検証 3: Example 2のトレース確認 
        # もし i2 が c1 に入ったなら、i1 (2番手) は弾かれる可能性がある
        # もし i2 が c2 に入ったなら、i3 (2番手) は弾かれる可能性がある
        # 最終的に {i2: c1, i3: c2} または {i1: c1, i2: c2} のような形になる
        if matching.get('i2') == 'c1':
            # c1はi2で埋まった。c2はi3(優先度2位)が入るはず (i1はc2適格だが最下位)
            assert matching.get('i3') == 'c2'
        elif matching.get('i2') == 'c2':
            # c2はi2で埋まった。c1はi1(優先度2位)が入るはず
            assert matching.get('i1') == 'c1'

    def test_respect_priorities_logic(self):
        """
        MMAの「追い出し(Swap)」ロジックの動作確認
        i1がc1を占有している状態で、より優先度の高いi2が登場した場合、
        i2がi1を追い出すことを確認。
        """
        agents = ['i1', 'i2']
        categories = ['c1']
        capacities = {'c1': 1}
        priorities = {'c1': ['i2', 'i1']}  # i2 > i1
        
        solver = MMASolver(agents, categories, capacities, priorities)
        
        # 内部メソッドをモック的に使って初期状態を強制する場合（クラス設計による）
        # ここではブラックボックステストとしてsolveを呼ぶ
        matching = solver.solve()
        
        assert matching['i2'] == 'c1'
        assert 'i1' not in matching