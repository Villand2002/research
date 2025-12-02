
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from scipy.stats import kendalltau
from codes.data_generation.mallows import generate_mallows_permutation, kendall_distance



def generate_priorities(agent_number: int, category_number: int, 
                        phi: float = 0.5, reference: List[int] = None) -> pd.DataFrame:
    """Mallows modelを使ってエージェントの選好を生成"""
    if reference is None:
        reference = list(range(category_number))
    
    preferences = []
    for i in range(agent_number):
        pref = generate_mallows_permutation(reference, phi)
        preferences.append(pref)
    
    df = pd.DataFrame(preferences, 
                      columns=[f'rank_{i+1}' for i in range(category_number)],
                      index=[f'agent_{i+1}' for i in range(agent_number)])
    return df

def generate_capacities(category_number: int, total_capacity: int = None,
                        distribution: str = 'uniform') -> pd.DataFrame:
    """カテゴリの受け入れ人数を生成"""
    if total_capacity is None:
        total_capacity = category_number * 10
    
    if distribution == 'uniform':
        base = total_capacity // category_number
        remainder = total_capacity % category_number
        capacities = [base] * category_number
        for i in range(remainder):
            capacities[i] += 1
    
    elif distribution == 'normal':
        raw = np.random.normal(loc=total_capacity/category_number, 
                               scale=total_capacity/(category_number*3), 
                               size=category_number)
        raw = np.maximum(raw, 1)
        capacities = (raw / raw.sum() * total_capacity).astype(int)
        diff = total_capacity - capacities.sum()
        capacities[0] += diff
    
    elif distribution == 'exponential':
        raw = np.random.exponential(scale=1, size=category_number)
        raw = np.sort(raw)[::-1]
        capacities = (raw / raw.sum() * total_capacity).astype(int)
        diff = total_capacity - capacities.sum()
        capacities[0] += diff
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    df = pd.DataFrame({
        'category': [f'category_{i}' for i in range(category_number)],
        'capacity': capacities
    })
    return df

def analyze_preferences(preferences_df: pd.DataFrame) -> dict:
    """選好の統計情報を計算"""
    n_agents, n_categories = preferences_df.shape
    rank_distribution = {}
    for rank_col in preferences_df.columns:
        rank_distribution[rank_col] = preferences_df[rank_col].value_counts().to_dict()
    
    reference = preferences_df.iloc[0].tolist()
    distances = []
    for i in range(1, n_agents):
        pref = preferences_df.iloc[i].tolist()
        dist = kendall_distance(reference, pref)
        distances.append(dist)
    
    return {
        'n_agents': n_agents,
        'n_categories': n_categories,
        'avg_kendall_distance': np.mean(distances) if distances else 0,
        'std_kendall_distance': np.std(distances) if distances else 0,
        'rank_distribution': rank_distribution
    }