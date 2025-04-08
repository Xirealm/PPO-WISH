import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict, deque
from agents.node_env import NodeOptimizationEnv

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        # 确保输入是2D张量 [batch_size, features]
        if x.dim() == 1:
            x = x.unsqueeze(0)  # 添加batch维度
        return self.network(x).squeeze(0)  # 如果是单个样本，移除batch维度

class NodeAgent:
    def __init__(self, env:NodeOptimizationEnv, alpha=0.1, gamma=0.9, epsilon_start=1.0, 
                 epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.memory = deque(maxlen=memory_size)
        self.best_memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        
        # DQN networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 5  # [feature_pos, feature_cross, physical_pos, physical_neg, physical_cross]
        self.action_dim = 4  # [remove_pos, remove_neg, restore_pos, restore_neg]
        
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)
        self.criterion = nn.SmoothL1Loss()
        
        self.best_pos = 100
        self.best_cross = 0
        self.best_pos_feature_distance = float('inf')
        self.best_cross_feature_distance = float('inf')

    def _get_state_features(self, state):
        # 提取图的特征作为状态向量
        feature_pos = np.mean(list(nx.get_edge_attributes(state, 'feature_pos').values())) if nx.get_edge_attributes(state, 'feature_pos') else 0
        feature_cross = np.mean(list(nx.get_edge_attributes(state, 'feature_cross').values())) if nx.get_edge_attributes(state, 'feature_cross') else 0
        physical_pos = np.mean(list(nx.get_edge_attributes(state, 'physical_pos').values())) if nx.get_edge_attributes(state, 'physical_pos') else 0
        physical_neg = np.mean(list(nx.get_edge_attributes(state, 'physical_neg').values())) if nx.get_edge_attributes(state, 'physical_neg') else 0
        physical_cross = np.mean(list(nx.get_edge_attributes(state, 'physical_cross').values())) if nx.get_edge_attributes(state, 'physical_cross') else 0
        
        return torch.tensor([feature_pos, feature_cross, physical_pos, physical_neg, physical_cross], 
                          device=self.device, dtype=torch.float32)

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_action(self, state):
        actions = self.get_possible_actions(state)
        if not actions:
            return None, None  # 无可用动作
            
        state_tensor = self._get_state_features(state)
        
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                # 只考虑可用动作的Q值
                action_mask = torch.zeros(self.action_dim, device=self.device)
                for act in actions:
                    act_type = act[1]
                    if "remove_pos" in act_type:
                        action_mask[0] = 1
                    elif "remove_neg" in act_type:
                        action_mask[1] = 1
                    elif "restore_pos" in act_type:
                        action_mask[2] = 1
                    elif "restore_neg" in act_type:
                        action_mask[3] = 1
                        
                q_values = q_values * action_mask - 1e9 * (1 - action_mask)
                action_idx = q_values.argmax().item()
                
                # 将action_idx映射回实际action
                for act in actions:
                    if ("remove_pos" in act[1] and action_idx == 0) or \
                       ("remove_neg" in act[1] and action_idx == 1) or \
                       ("restore_pos" in act[1] and action_idx == 2) or \
                       ("restore_neg" in act[1] and action_idx == 3):
                        action = act
                        break
        print(f"Selected action: {action}")
        return action

    def get_possible_actions(self, state):
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

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        
        state_batch = torch.stack([t[0] for t in transitions])
        action_batch = torch.tensor([[t[1][1].startswith("remove_pos"),
                                    t[1][1].startswith("remove_neg"),
                                    t[1][1].startswith("restore_pos"),
                                    t[1][1].startswith("restore_neg")] 
                                   for t in transitions], device=self.device).float()
        reward_batch = torch.tensor([t[2] for t in transitions], device=self.device)
        next_state_batch = torch.stack([t[3] for t in transitions])
        
        current_q_values = (self.policy_net(state_batch) * action_batch).sum(dim=1)
        
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
        
        expected_q_values = reward_batch + self.gamma * next_q_values
        
        loss = self.criterion(current_q_values, expected_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def replay(self):
        self.optimize()

    def replay_best(self):
        if len(self.best_memory) < self.batch_size:
            return
        old_memory = self.memory
        self.memory = self.best_memory
        self.optimize()
        self.memory = old_memory