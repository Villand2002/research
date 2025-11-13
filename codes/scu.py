import networkx as nx
from typing import Dict, List, Tuple


class Agent:
    def __init__(self, agent_id: int, acceptable_categories: List[int]):
        self.agent_id = agent_id
        self.acceptable_categories = acceptable_categories

    def show_info(self):
        print(f"Agent ID: {self.agent_id}, Acceptable Categories: {self.acceptable_categories}")
        
class Agents:
    def __init__(self, agents: List[Agent], agent_number: int):
        self.agents = agents
        self.agent_number = agent_number

    def show_all_info(self):
        for agent in self.agents:
            agent.show_info()

class Category:
    def __init__(self, category_id: int, capacity: int, priority: List[int]):
        self.category_id = category_id
        self.capacity = capacity
        self.priority = priority
        self.eligible_agents = []  # ← 後で自動生成する

    def show_info(self):
        print(f"Category ID: {self.category_id}, Capacity: {self.capacity}, Priority: {self.priority}, Eligible: {self.eligible_agents}")


def assign_eligibility(agents_obj, categories_obj):
    """
    Agent.acceptable_categories を元に Category.eligible_agents を生成する
    """
    for cat in categories_obj:
        cat.eligible_agents = []

    for ag in agents_obj.agents:
        for c in ag.acceptable_categories:
            for cat in categories_obj:
                if cat.category_id == c:
                    cat.eligible_agents.append(ag.agent_id)



def compute_maximum_matching(agents_obj, categories_obj, fixed_assignments=None):
    G = nx.Graph()

    if fixed_assignments is None:
        fixed_assignments = {}

    # eligibility edges を作る
    for ag in agents_obj.agents:
        a_id = ag.agent_id

        if a_id in fixed_assignments:
            continue

        for cat in categories_obj:
            if a_id in cat.eligible_agents:
                G.add_edge(f"A_{a_id}", f"C_{cat.category_id}")

    # 最大マッチング
    match = nx.algorithms.matching.max_weight_matching(G, maxcardinality=True)

    mu = dict(fixed_assignments)

    for x, y in match:
        if x.startswith("A_"):
            a = int(x[2:])
            c = int(y[2:])
        else:
            a = int(y[2:])
            c = int(x[2:])
        mu[a] = c

    return mu
def execute_scu(agents_obj, categories_obj, category_order):
    """
    SCU Algorithm Implementation
    category_order: 論文の ⊳ の strict order のリスト (desc)
    """
    
    # eligibility 自動生成
    assign_eligibility(agents_obj, categories_obj)

    # 初期 μ を最大マッチングで作る
    mu = compute_maximum_matching(agents_obj, categories_obj)

    X = set()   # 確定 agent
    Y = set()   # 処理済カテゴリ

    # categories を辞書化して扱いやすくする
    cat_map = {c.category_id: c for c in categories_obj}

    for c in category_order:
        if c in Y:
            continue

        cat = cat_map[c]

        while True:
            assigned_to_c = [i for i, cc in mu.items() if cc == c and i in X]

            if len(assigned_to_c) >= cat.capacity:
                break

            candidates = [
                ag.agent_id for ag in agents_obj.agents
                if ag.agent_id not in X and ag.agent_id in cat.eligible_agents
            ]

            if not candidates:
                break

            # priority に基づき昇順（小さい index = priority 高）
            candidates.sort(key=lambda i: cat.priority.index(i))

            updated = False

            for i in candidates:
                mu_prime = dict(mu)
                mu_prime[i] = c

                mu_check = compute_maximum_matching(
                    agents_obj, categories_obj,
                    fixed_assignments={i: c}
                )

                if len(mu_check) == len(mu_prime):
                    mu = mu_prime
                    X.add(i)
                    updated = True
                    break

            if not updated:
                break

        Y.add(c)

    return list(mu.items())
