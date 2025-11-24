from agent import Agent, Agents, Category
from generate_data import generate_priorities, analyze_preferences, generate_capacities
from graph_function import assign_eligibility_random, assign_category_priorities
from algorithm import rev_algorithm, execute_mma

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