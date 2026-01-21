import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.algorithm.scu_comb import SCUcomb


class TestSCUcomb:
    def test_example4_paper(self):
        """
        Paper Example 4
        SCUが以下を満たすかテスト:
        1. Maximum Cardinality (m=3)
        2. Maximum Beneficiary Assignment (b=2) for {c1, c3}
        3. Precedence Order (c1 > c2 > c3) の遵守
        """
        agents = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']
        categories = ['c1', 'c2', 'c3']
        capacities = {'c1': 1, 'c2': 1, 'c3': 1}

        priorities = {
            'c1': ['i2', 'i1', 'i3'],
            'c2': ['i2', 'i4', 'i6', 'i3', 'i5', 'i1'],
            'c3': ['i5', 'i4', 'i6']
        }

        precedence_order = ['c1', 'c2', 'c3']
        beneficial_categories = ['c1', 'c3']

        solver = SCUcomb(
            agents, categories, capacities, priorities,
            precedence_order, beneficial_categories
        )
        matching = solver.solve()

        assert len(matching) == 3

        beneficial_matches = [c for c in matching.values() if c in beneficial_categories]
        assert len(beneficial_matches) == 2
        assert 'c1' in matching.values()
        assert 'c3' in matching.values()

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
        beneficial = ['c1']

        solver = SCUcomb(agents, categories, capacities, priorities, precedence, beneficial)

        b = solver._compute_max_beneficial_compact()
        m = solver._compute_max_cardinality_compact(b)
        assert b == 1
        assert m == 2
