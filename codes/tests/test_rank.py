import unittest
from algorithm.rank import RankSolver

class TestRankMechanism(unittest.TestCase):
    def setUp(self):
        # 論文 Page 40-41, Example 6 のデータ
        self.agents = [1, 2, 3, 4]
        # エージェントの希望（Rank）リスト
        self.agent_ranks = {
            1: [2, 1], # d2, d1
            2: [1, 2], # d1, d2
            3: [1, 3], # d1, d3
            4: [3]     # d3
        }
        # 病院（カテゴリ）側のデータ
        self.cat_dicts = [
            {'id': 1, 'capacity': 1, 'priority': [1, 3, 2]}, # d1: 1, 3, 2
            {'id': 2, 'capacity': 1, 'priority': [2, 1]},    # d2: 2, 1
            {'id': 3, 'capacity': 1, 'priority': [4, 3]}     # d3: 4, 3
        ]

    def test_rank_logic_example6(self):
        solver = RankSolver(self.agents, self.cat_dicts)
        matching = solver.solve(self.agent_ranks)
        
        # 1. 最大サイズの検証 (定員合計3に対し、全員適格者がいるので3人がマッチするはず)
        self.assertEqual(len(matching), 3)
        
        # 2. 基本的な適格性の確認
        for a_id, c_id in matching.items():
            self.assertIn(c_id, self.agent_ranks[a_id]) # 希望リストに入っているか
            cat_info = next(c for c in self.cat_dicts if c['id'] == c_id)
            self.assertIn(a_id, cat_info['priority']) # 優先順位リスト（適格者）に入っているか

        print(f"Rank Matching Result (Example 6): {matching}")

if __name__ == "__main__":
    unittest.main()