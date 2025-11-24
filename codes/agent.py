from typing import List

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
        # アルゴリズムで使用する動的プロパティ
        self.eligible_agents: List[int] = [] 

    def show_info(self):
        print(f"Category ID: {self.category_id}, Capacity: {self.capacity}, Priority: {self.priority}")