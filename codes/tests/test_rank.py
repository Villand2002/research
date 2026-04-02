import unittest

class TestRankMechanismWithActualClass(unittest.TestCase):
    def setUp(self):
        # 論文 Example 6 (DA is bossy) のデータを再現 
        # エージェントの定義 (acceptable_categories に順位を込める)
        self.agent_list = [
            Agent(agent_id=1, acceptable_categories=[2, 1]), # d2, d1
            Agent(agent_id=2, acceptable_categories=[1, 2]), # d1, d2
            Agent(agent_id=3, acceptable_categories=[1, 3]), # d1, d3
            Agent(agent_id=4, acceptable_categories=[3])     # d3
        ]
        self.agent_ids = [a.agent_id for a in self.agent_list]

        # カテゴリ（病院）の定義 
        self.cat_dicts = [
            {'id': 1, 'capacity': 1, 'priority': [1, 3, 2]}, # d1
            {'id': 2, 'capacity': 1, 'priority': [2, 1]},    # d2
            {'id': 3, 'capacity': 1, 'priority': [4, 3]}     # d3
        ]

    def test_rank_behavior(self):
        # 選好リストを取得
        agent_ranks = {a.agent_id: a.acceptable_categories for a in self.agent_list}
        
        solver = RankSolver(self.agent_ids, self.cat_dicts)
        matching = solver.solve(agent_ranks)

        # 論文の結論: エージェント最適マッチングは d1-1, d2-2, d3-4 
        # Rankメカニズム（最大重み最大マッチング）の結果を確認
        self.assertEqual(matching.get(1), 1) # Agent 1 -> Category 1
        self.assertEqual(matching.get(2), 2) # Agent 2 -> Category 2
        self.assertEqual(matching.get(4), 3) # Agent 4 -> Category 3
        
        # マッチングのサイズの確認 (最大マッチングであること)
        self.assertEqual(len(matching), 3)

if __name__ == "__main__":
    unittest.main()