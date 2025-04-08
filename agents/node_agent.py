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
            nn.LayerNorm(256),  
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 512),
            nn.LayerNorm(512), 
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256),  
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, action_dim)
        )
        
    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        out = self.network(x)
        return out.squeeze(0) if out.size(0) == 1 else out

class NodeAgent:
    def __init__(self, env:NodeOptimizationEnv, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64):
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
        
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.criterion = nn.SmoothL1Loss()  # 添加这一行
        
        self.best_pos = 100
        self.best_cross = 0
        self.best_pos_feature_distance = float('inf')
        self.best_cross_feature_distance = float('inf')
        
        self.max_nodes = 1600  # 添加最大节点数限制
        self.action_dim = self.max_nodes * 4  # 固定动作空间大小

    def initialize_networks(self, features):
        """根据特征张量初始化网络"""
        if self.policy_net is not None:
            return  # 如果网络已经初始化，直接返回 
            
        # 使用固定的action_dim
        action_dim = self.action_dim
        
        # 初始化网络
        self.policy_net = DQN(self.state_dim, action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

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
        # 确保网络已初始化
        if self.policy_net is None:
            features = self.env.features
            self.initialize_networks(features)
        
        actions = self.get_possible_actions(state)
        if not actions:
            return None, None  # 无可用动作
            
        state_tensor = self._get_state_features(state)
        
        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                # 创建动作掩码，大小与q_values匹配
                action_mask = torch.zeros_like(q_values, device=self.device)
                # 只为实际存在的节点设置掩码
                valid_nodes = list(self.env.original_G.nodes())
                # 为每个可能的动作设置掩码
                for act in actions:
                    node_idx = valid_nodes.index(act[0])
                    if node_idx >= self.max_nodes:
                        continue  # 跳过超出范围的节点
                    if "remove_pos" in act[1]:
                        action_mask[node_idx * 4 + 0] = 1
                    elif "remove_neg" in act[1]:
                        action_mask[node_idx * 4 + 1] = 1
                    elif "restore_pos" in act[1]:
                        action_mask[node_idx * 4 + 2] = 1
                    elif "restore_neg" in act[1]:
                        action_mask[node_idx * 4 + 3] = 1
                
                q_values = q_values * action_mask - 1e9 * (1 - action_mask)
                action_idx = q_values.argmax().item()
                
                # 将action_idx映射回实际action
                node_idx = action_idx // 4
                action_type = action_idx % 4
                node = list(self.env.original_G.nodes())[node_idx]
                
                action_types = ["remove_pos", "remove_neg", "restore_pos", "restore_neg"]
                target_type = action_types[action_type]
                
                # 在可能的动作中找到对应的动作
                for act in actions:
                    if act[0] == node and target_type in act[1]:
                        action = act
                        break
                else:
                    action = random.choice(actions)  # 如果没找到对应动作，随机选择一个

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
        if self.policy_net is None or len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        action_dim = self.policy_net.network[-1].out_features  # 从网络获取当前action_dim
        
        state_batch = torch.stack([t[0] for t in transitions])
        action_batch = torch.zeros(self.batch_size, action_dim, device=self.device)
        for i, t in enumerate(transitions):
            node_idx = t[1][0]
            if node_idx >= self.max_nodes:
                continue  # 跳过超出范围的节点
            if t[1][1].startswith("remove_pos"):
                action_batch[i, node_idx * 4 + 0] = 1
            elif t[1][1].startswith("remove_neg"):
                action_batch[i, node_idx * 4 + 1] = 1
            elif t[1][1].startswith("restore_pos"):
                action_batch[i, node_idx * 4 + 2] = 1
            elif t[1][1].startswith("restore_neg"):
                action_batch[i, node_idx * 4 + 3] = 1
                
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