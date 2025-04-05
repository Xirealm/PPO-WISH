import random
import numpy as np
import networkx as nx
from collections import defaultdict, deque
from agents.node_env import NodeOptimizationEnv

class NodeAgent:
    def __init__(self, env:NodeOptimizationEnv, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64,reward_threshold=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.q_table = defaultdict(float)
        self.best_pos = 100
        self.best_cross = 0
        self.best_q_table = None
        self.memory = deque(maxlen=memory_size)
        self.best_memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.reward_threshold = reward_threshold
        self.last_reward = None
        self.best_reward = -float('inf')
        self.best_reward_save = -float('inf')
        self.best_pos_feature_distance = float('inf')
        self.best_cross_feature_distance = float('inf')

    def update_epsilon(self):
        """Update the epsilon value based on a decay factor."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _get_state_hash(self, state):
        # 提取图的关键特征作为状态
        pos_nodes = sorted([n for n, d in state.nodes(data=True) if d['category'] == 'pos'])
        neg_nodes = sorted([n for n, d in state.nodes(data=True) if d['category'] == 'neg'])
        
        # 计算特征距离的平均值
        feature_pos = np.mean(list(nx.get_edge_attributes(state, 'feature_pos').values())) if nx.get_edge_attributes(state, 'feature_pos') else 0
        feature_cross = np.mean(list(nx.get_edge_attributes(state, 'feature_cross').values())) if nx.get_edge_attributes(state, 'feature_cross') else 0
        
        # 组合状态特征
        state_tuple = (
            tuple(pos_nodes),
            tuple(neg_nodes),
            round(feature_pos, 4),
            round(feature_cross, 4)
        )
        return hash(state_tuple)

    def get_action(self, state):
        """Select an action based on epsilon-greedy policy."""
        actions = self.get_possible_actions(state)

        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            # 使用哈希后的状态查询Q值
            state_hash = self._get_state_hash(state)
            q_values = {action: self.q_table.get((state_hash, action), 0) for action in actions}
            max_q = max(q_values.values())
            action = random.choice([action for action, q in q_values.items() if q == max_q])  

        return action

    def get_possible_actions(self, state):
        """Get the possible actions for the current state."""
        actions = []

        restore_pos_actions = [
            (node, "restore_pos")
            for node in self.env.removed_nodes
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'pos'
        ]
        restore_neg_actions = [
            (node, "restore_neg")
            for node in self.env.removed_nodes
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'neg'
        ]
        remove_pos_actions = [
            (node, "remove_pos")
            for node in state.nodes()
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'pos'
        ]
        remove_neg_actions = [
            (node, "remove_neg")
            for node in state.nodes()
            if node in self.env.original_G and self.env.original_G.nodes[node].get('category') == 'neg'
        ]

        pos_nodes_count = len(self.env.pos_nodes)
        neg_nodes_count = len(self.env.neg_nodes)

        if self.env.min_nodes < pos_nodes_count < self.env.max_nodes and self.env.min_nodes < neg_nodes_count < self.env.max_nodes:
            actions.extend(restore_pos_actions)
            actions.extend(restore_neg_actions)
            actions.extend(remove_pos_actions)
            actions.extend(remove_neg_actions)
            # actions.extend([(node, "add") for node in range(state.number_of_nodes(), state.number_of_nodes() + 10)])
        else:
            if pos_nodes_count <= self.env.min_nodes:
                actions.extend(restore_pos_actions)
            elif pos_nodes_count >= self.env.max_nodes:
                actions.extend(remove_pos_actions)

            if neg_nodes_count <= self.env.min_nodes:
                actions.extend(restore_neg_actions)
            elif neg_nodes_count >= self.env.max_nodes:
                actions.extend(remove_neg_actions)

        return actions

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table based on the current state, action, reward, and next state."""
        state_hash = self._get_state_hash(state)
        next_state_hash = self._get_state_hash(next_state)
        
        max_next_q = max(
            [self.q_table.get((next_state_hash, next_action), 0) for next_action in self.get_possible_actions(next_state)],
            default=0)
        old_q = self.q_table.get((state_hash, action), 0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state_hash, action)] = new_q

    def replay(self):
        """Replay experiences from memory to update Q-table."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)

    def replay_best(self):
        """Replay best experiences from memory to update Q-table."""
        if len(self.best_memory) < self.batch_size:
            return

        batch = random.sample(self.best_memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state) 