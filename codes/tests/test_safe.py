import unittest
import networkx as nx
from typing import Dict, List

# 前述のクラス・関数が定義されている前提でテストを書きます
# from your_module import MMASolver, SafeSolver

class TestMatchingMechanisms(unittest.TestCase):
    
    def setUp(self):
        """
        論文の構成要素に基づいた共通データのセットアップ
        Example 1付近の構造をモデル化 [cite: 3]
        """
        self.agents = [1, 2, 3, 4]
        self.categories = ['c1', 'c2']
        self.capacities = {'c1': 1, 'c2': 1}
        # 優先順位リスト（先頭ほど優先度が高い）
        self.priorities = {
            'c1': [2, 3, 1], # エージェント2が最優先 [cite: 2631]
            'c2': [2, 1]     # エージェント3はc2に不適格とする例 [cite: 2631]
        }


    def test_safe_mechanism_strategy(self):
        """
        Safeメカニズムのテスト
        エージェントの「選択（Choice）」に基づき、各枠が独立して処理されることを確認 [cite: 2880, 2883]
        """
        # エージェントの選択を定義
        # A1はc1を、A2はc2を選択したとする
        agent_choices = {1: 'c1', 2: 'c2', 3: 'c1'}
        
        # 内部用カテゴリデータ
        cat_dicts = [
            {'id': 'c1', 'capacity': 1, 'priority': [2, 3, 1]},
            {'id': 'c2', 'capacity': 1, 'priority': [2, 1]}
        ]
        
        from codes.algorithm.safe import SafeSolver # 定義済みとする
        solver = SafeSolver(self.agents, cat_dicts)
        matching = solver.solve(agent_choices)
        
        # 1. カテゴリc1の検証
        # c1を選択したのは1と3。優先順位は3 > 1なので、3が選ばれるべき
        self.assertEqual(matching.get(3), 'c1')
        self.assertNotIn(1, matching) # 1は落選
        
        # 2. カテゴリc2の検証
        # c2を選択したのは2のみ。適格なので2が選ばれる
        self.assertEqual(matching.get(2), 'c2')
        
        print(f"Safe Matching Result: {matching}")

    def test_feasibility_verification(self):
        """
        適格性（Eligibility）の検証テスト
        不適格なエージェントがマッチングされていないか確認 [cite: 2593]
        """
        # エージェント4はどのカテゴリの優先順位リストにも入っていない（不適格）
        solver = MMASolver(self.agents, self.categories, self.capacities, self.priorities)
        matching = solver.solve()
        
        self.assertNotIn(4, matching.keys())
        for agent, cat in matching.items():
            self.assertIn(agent, self.priorities[cat])

if __name__ == "__main__":
    unittest.main()