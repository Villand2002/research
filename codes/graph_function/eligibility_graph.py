import numpy as np
from typing import List
from codes.agent import Agents, Category
from codes.data_generation.mallows import generate_mallows_permutation

def assign_eligibility_random(agents_obj: Agents, categories_obj: List[Category], p=0.4):
    """Agents と Category の構造に合わせて eligibility を設定"""
    for cat in categories_obj:
        eligible = []
        for ag in agents_obj.agents:
            if np.random.rand() < p:
                eligible.append(ag.agent_id)
        if len(eligible) == 0:
            eligible.append(np.random.choice([ag.agent_id for ag in agents_obj.agents]))
        cat.eligible_agents = eligible

def assign_category_priorities(agents_obj: Agents, categories_obj: List[Category], phi=0.6):
    """各カテゴリに対して agent の優先順位を Mallows で生成し、Categoryオブジェクトにセット"""
    reference = [ag.agent_id for ag in agents_obj.agents]
    for cat in categories_obj:
        perm = generate_mallows_permutation(reference, phi)
        cat.priority = perm

