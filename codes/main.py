from agent import Agent, Agents, Category
from algorithm.mma import MMASolver, execute_mma
from algorithm.rev import rev_algorithm
from algorithm.scu import SCUSolver
from codes.data_generation.mallows import (
    analyze_preferences,
    generate_capacities,
    generate_priorities,
)
from codes.graph_function.eligibility_graph import (
    assign_category_priorities,
    assign_eligibility_random,
)

if __name__ == "__main__":
    # --- パラメータ設定 ---
    n_agents = 10
    n_categories = 5
    phi = 0.7
    
    # --- データ生成 ---
    print("=" * 50)
    print("選好データの生成")
    print("=" * 50)
    preferences = generate_priorities(n_agents, n_categories, phi=phi)
    stats = analyze_preferences(preferences)
    print(f"平均Kendall距離: {stats['avg_kendall_distance']:.2f}")
    
    total_capacity = 80
    capacities_uniform = generate_capacities(n_categories, total_capacity, 'uniform')
    
    # CSV保存 (必要であれば)
    # preferences.to_csv('preferences.csv')
    # capacities_uniform.to_csv('capacities.csv', index=False)

    # --- オブジェクト構築 ---
    cap_dict = {i: capacities_uniform.loc[i, "capacity"] for i in range(n_categories)}
    
    categories_obj = [
        Category(category_id=i, capacity=cap_dict[i], priority=[])
        for i in range(n_categories)
    ]
    
    agents_obj = Agents(
        agents=[Agent(agent_id=i, acceptable_categories=[]) for i in range(1, n_agents+1)],
        agent_number=n_agents
    )

    # --- グラフ用設定 (Eligibility & Priority) ---
    assign_eligibility_random(agents_obj, categories_obj, p=0.3)
    assign_category_priorities(agents_obj, categories_obj, phi=0.6)

    # --- アルゴリズム実行: REV ---
    print("\nRunning REV Algorithm...")
    matching, rejected = rev_algorithm(agents_obj, categories_obj)
    print(f"Rejected agents: {rejected}")
    print("Matching result (Top 5):")
    for a, c in matching[:5]:
        print(f"Agent {a} -> Category {c}")
        
    # --- アルゴリズム実行: MMA ---
    print("\nRunning MMA Algorithm...")
    mma_matching = execute_mma(agents_obj, categories_obj)
    print("Matching result (Top 5):")
    for a, c in mma_matching[:5]:
        print(f"Agent {a} -> Category {c}")


# MMAの例
    agents = ['i1', 'i2', 'i3']
    categories = ['c1', 'c2']
    capacities = {'c1': 1, 'c2': 1}
    
    # 優先順位 (左が優先度高)
    # c1: i2 > i1 > empty > i3 (i3は適格外とするか、リスト末尾とする。ここでは適格リストとして定義)
    # c2: i2 > i3 > empty > i1
    priorities = {
        'c1': ['i2', 'i1'],       # i3は不適格(emptyより下)と仮定
        'c2': ['i2', 'i3', 'i1']  # i1は不適格の可能性もあるが、例として含める
    }
    
    # ※論文のFig 1を見ると、c1に対してi3は適格ではない(edgesがない)。c2に対してi1は適格ではない。
    # グラフの定義に合わせて修正します [cite: 103]
    priorities_corrected = {
        'c1': ['i2', 'i1'], 
        'c2': ['i2', 'i3']
    }

    mma = MMASolver(agents, categories, capacities, priorities_corrected)
    result = mma.solve()
    
    print("\nFinal Matching:")
    for agent, cat in sorted(result.items()):
        print(f"{agent} -> {cat}")
        
        
        
    # Example 4 Settings
    agents = ['i1', 'i2', 'i3', 'i4', 'i5', 'i6']
    categories = ['c1', 'c2', 'c3']
    
    # Beneficial: c1, c3
    # Open: c2
    beneficial = ['c1', 'c3']
    
    # Precedence: c1 > c2 > c3
    precedence = ['c1', 'c2', 'c3']
    
    capacities = {'c1': 1, 'c2': 1, 'c3': 1}
    
    # Priorities (論文 p.18)
    # c1: i2 > i1 > i3
    # c2: i2 > i4 > i6 > i3 > i5 > i1
    # c3: i5 > i4 > i6
    priorities = {
        'c1': ['i2', 'i1', 'i3'],
        'c2': ['i2', 'i4', 'i6', 'i3', 'i5', 'i1'],
        'c3': ['i5', 'i4', 'i6']
    }

    print("--- SCU Algorithm (Example 4) ---")
    scu = SCUSolver(agents, categories, capacities, priorities, precedence, beneficial)
    result = scu.solve()
    
    # 結果表示
    print("\nFinal Matching:")
    for a in sorted(result.keys()):
        print(f"{a} -> {result[a]}")