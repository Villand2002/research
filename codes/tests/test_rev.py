import sys
from pathlib import Path

import pytest

# プロジェクトルートをパスに追加
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from codes.agent import Agent, Agents, Category
from codes.algorithm.rev import rev_algorithm_flow_network


class TestREVAlgorithm:
    @pytest.fixture
    def setup_example1_objects(self):
        """
        [cite_start]論文 Example 1 [cite: 94-102] のデータセットアップ
        
        Agents: {1, 2, 3}
        Categories: {1, 2} (Capacity 1)
        Priorities:
            c1: 2 > 1  (3は適格外)
            c2: 2 > 3  (1は適格外)
        """
        # 1. エージェントの作成
        # idはintで管理されている想定 (rev_algorithm内の ag.agent_id の使用法より)
        a1 = Agent(agent_id=1, acceptable_categories=[1]) # c2は不適格なので含めない想定だが、REVのロジックはCategory側のeligible_agentsを見る
        a2 = Agent(agent_id=2, acceptable_categories=[1, 2])
        a3 = Agent(agent_id=3, acceptable_categories=[2])
        
        agents_obj = Agents(agents=[a1, a2, a3], agent_number=3)

        # 2. カテゴリの作成
        # Category(category_id, capacity, priority)
        c1 = Category(category_id=1, capacity=1, priority=[2, 1])
        c2 = Category(category_id=2, capacity=1, priority=[2, 3])

        # 3. Eligible Agentsの設定
        # graph_function.py の compute_max_matching_size は cat.eligible_agents を参照するため手動設定
        c1.eligible_agents = [2, 1]
        c2.eligible_agents = [2, 3]

        categories_obj = [c1, c2]

        return agents_obj, categories_obj

    def test_rev_example1(self, setup_example1_objects):
        """
        Example 1 に対する REV アルゴリズムのテスト
        """
        agents_obj, categories_obj = setup_example1_objects

        # アルゴリズム実行
        matching, banned = rev_algorithm_flow_network(agents_obj, categories_obj)

        # --- 検証 ---
        
        # 1. Matching形式の変換 [(agent_id, category_id), ...] -> dict
        matching_dict = {m[0]: m[1] for m in matching}

        # 2. 最大基数 (Maximum Cardinality) の確認
        # Example 1では、最大2人がマッチ可能 (例: {2->c1, 3->c2} や {1->c1, 2->c2})
        assert len(matching) == 2
        
        # 3. 優先順位の尊重 (Respect for Priorities) & 強者の保護
        # Agent 2 は c1, c2 両方でトップ優先順位なので、必ずマッチするはず
        assert 2 in matching_dict
        
        # 4. 誰がマッチしたかの詳細確認
        # REVは「Baseline Orderの逆順」に「削除しても最大マッチング数が減らない弱者」をBanする
        # ここで Baseline は agent_id 昇順 [1, 2, 3] とする
        # 逆順チェック: 3 -> 2 -> 1
        
        # Check 3: 3をBanしても、{1, 2} でマッチング数2 ({1->c1, 2->c2}) が作れる -> 3はBanされる可能性あり？
        # しかし、2をBanすると、{1, 3} でマッチング数2 ({1->c1, 3->c2}) が作れる。
        
        # 実際のREVの挙動(強者が残る)を確認
        print(f"\nREV Result Matches: {matching_dict}")
        print(f"Banned Agents: {banned}")

        # c1, c2ともに定員1なので合計2枠。Agentは3人。1人あぶれる。
        assert len(banned) == 1
