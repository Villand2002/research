import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.algorithm.scu import SCUSolver
from codes.agent import Agent, Agents, Category
from codes.data_generation.generate_data import generate_priorities, generate_capacities
from codes.graph_funtion import compute_max_matching_size, nx_rebuild_graph

class TestSCUSolver:
    def test_example4_paper(self):
        """
        Paper Example 4 
        SCUが以下を満たすかテスト:
        1. Maximum Cardinality (m=3)
        2. Maximum Beneficiary Assignment (b=2) for {c1, c3}
        3. Precedence Order (c1 > c2 > c3) の遵守
        """
        # データセットアップ 
        agents = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']
        categories = ['c1', 'c2', 'c3']
        capacities = {'c1': 1, 'c2': 1, 'c3': 1}
        
        # Priorities
        # c1: i2 > i1 > i3
        # c2: i2 > i4 > i6 > i3 > i5 > i1
        # c3: i5 > i4 > i6
        priorities = {
            'c1': ['i2', 'i1', 'i3'],
            'c2': ['i2', 'i4', 'i6', 'i3', 'i5', 'i1'],
            'c3': ['i5', 'i4', 'i6']
        }
        
        precedence_order = ['c1', 'c2', 'c3']
        beneficial_categories = ['c1', 'c3']
        
        solver = SCUSolver(
            agents, categories, capacities, priorities, 
            precedence_order, beneficial_categories
        )
        matching = solver.solve()
        
        # 検証 1: Maximum Cardinality [cite: 125, 440]
        # 全カテゴリ容量が3で、適格エージェントは十分いるため、3枠埋まるはず
        assert len(matching) == 3
        
        # 検証 2: Maximum Beneficiary Assignment [cite: 264, 440]
        # Beneficial (c1, c3) は両方埋まるはず (b=2)
        beneficial_matches = [c for c in matching.values() if c in beneficial_categories]
        assert len(beneficial_matches) == 2
        assert 'c1' in matching.values()
        assert 'c3' in matching.values()
        
        # 検証 3: 具体的なマッチング結果の妥当性
        # Precedence c1 > c2 > c3 なので c1 の決定が優先される
        # c1のトップ i2 は c1 に入る可能性が高い
        if matching.get('i2') == 'c1':
            # c1=i2 の場合
            # 残り c2, c3。Beneficialである c3 を埋める必要がある
            # c3の候補: i5, i4, i6
            # c2の候補: i4, i6, i3, i5, i1
            
            # もし i5 が c3 に入ると、c2 は i4 あたりが入る -> {i2:c1, i4:c2, i5:c3}
            # これはValid
            pass
        
        # c1 に i2 が入っていない場合 (SCUのロジック上次善のマッチングが選ばれた場合)
        # i2 が c2 に入り、c1 に i1 が入るパターンもあり得るが、
        # c1 > c2 (Precedence) なので i2 は c1 を優先するはず
        assert matching.get('i2') == 'c1', "Higher precedence category c1 should take highest priority agent i2"

    def test_scu_invariants(self):
        """
        内部ロジックテスト: b (Beneficial数) と m (Total数) の計算
        """
        agents = ['i1', 'i2']
        categories = ['c1', 'c2']
        capacities = {'c1': 1, 'c2': 1}
        priorities = {
            'c1': ['i1'],
            'c2': ['i2']
        }
        precedence = ['c1', 'c2']
        beneficial = ['c1'] # c1のみBeneficial
        
        solver = SCUSolver(agents, categories, capacities, priorities, precedence, beneficial)
        
        # Solver内部の計算メソッドをテスト (実装依存)
        # もし _calc_max_values があるなら
        if hasattr(solver, '_calc_max_values'):
            b, m = solver._calc_max_values()
            assert b == 1 # c1が埋まる
            assert m == 2 # c1, c2両方埋まる