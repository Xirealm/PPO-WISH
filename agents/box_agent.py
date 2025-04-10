import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from agents.box_env import BoxOptimizationEnv

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

class BoxAgent:
    def __init__(self, env:BoxOptimizationEnv, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        # 首先定义动作空间
        self.edge_actions = ["shrink_up", "shrink_down", "shrink_left", "shrink_right","expand_up", "expand_down", "expand_left", "expand_right"]
        self.merge_action = ["merge"]
        self.actions = self.edge_actions + self.merge_action
        
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        
        # DQN networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = 6  # [box_features, coherence, difference, adjacency, pos_ratio, neg_ratio] 
        self.action_dim = len(self.edge_actions) + len(self.merge_action)
        
        # 初始化为None
        self.policy_net = None
        self.target_net = None
        self.optimizer = None
        self.criterion = nn.SmoothL1Loss()
        
        self.memory = deque(maxlen=10000)
        self.best_memory = deque(maxlen=10000)
        self.batch_size = 64

    def initialize_networks(self, features):
        """根据特征张量初始化网络"""
        if self.policy_net is not None:
            return  # 如果已经初始化过则直接返回
            
        self.policy_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net = DQN(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.alpha)

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def _get_state_features(self, state):
        G, boxes = state
        box_features = []
        for box_idx in range(len(boxes)):
            features = self.env.get_box_features(box_idx)
            if features is not None:
                box_features.append(torch.mean(features))
            
        box_features_mean = torch.mean(torch.stack(box_features)) if box_features else torch.tensor(0.)
        coherence = self.env.calculate_inner_coherence()
        difference = self.env.calculate_outer_difference()
        
        # Calculate other metrics
        pos_ratio = self.env.calculate_pos_ratio_in_box()
        neg_ratio = self.env.calculate_neg_ratio_out_box()
        
        # Get adjacency info
        adjacency = len([1 for box_idx in range(len(boxes)) 
                        if any(self.env.check_box_adjacency(box_idx).values())])
        
        return torch.tensor([box_features_mean, coherence, difference,
                           adjacency/len(boxes), pos_ratio, neg_ratio],
                          device=self.device, dtype=torch.float32)

    def get_action(self, state):
        # 确保网络已初始化
        if self.policy_net is None:
            features = self.env.features
            self.initialize_networks(features)
            
        state_tensor = self._get_state_features(state)
        
        if random.random() < self.epsilon:
            action_type = random.choice(self.actions)
            if action_type in self.edge_actions:
                # 根据动作类型选择合适的box
                if action_type in ["shrink_up", "expand_up"]:
                    # 选择y坐标最小的box
                    box_idx = min(range(len(self.env.boxes)), 
                        key=lambda i: self.env.boxes[i]['min_y'])
                elif action_type in ["shrink_down", "expand_down"]:
                    # 选择y坐标最大的box
                    box_idx = max(range(len(self.env.boxes)), 
                        key=lambda i: self.env.boxes[i]['max_y'])
                elif action_type in ["shrink_left", "expand_left"]:
                    # 选择x坐标最小的box
                    box_idx = min(range(len(self.env.boxes)), 
                        key=lambda i: self.env.boxes[i]['min_x'])
                elif action_type in ["shrink_right", "expand_right"]:
                    # 选择x坐标最大的box
                    box_idx = max(range(len(self.env.boxes)), 
                        key=lambda i: self.env.boxes[i]['max_x'])
                return (box_idx, action_type, None)
            else:
                # merge操作保持不变
                operation_type = random.choice(['extreme', 'adjacent'])
                extreme_boxes = []
                adjacent_boxes = []
                
                # 分类所有box
                for box_idx in range(len(self.env.boxes)):
                    is_extreme, all_adjacent, free_edges = self.get_box_type(box_idx)
                    if is_extreme:
                        extreme_boxes.append((box_idx, free_edges))
                    if all_adjacent:
                        adjacent_boxes.append(box_idx)
                
                if operation_type == 'extreme' and extreme_boxes:
                    # 随机选择一个极值box
                    box_idx, free_edges = random.choice(extreme_boxes)
                    # 在可用的方向中随机选择一个
                    available_directions = [d for d, is_free in free_edges.items() if is_free]
                    if available_directions:
                        direction = random.choice(available_directions)
                        operation = random.choice([f"shrink_{direction}", f"expand_{direction}"])
                        return (box_idx, operation, None)
                        
                elif operation_type == 'adjacent' and adjacent_boxes:
                    # 随机选择一个四面邻接的box
                    box_idx = random.choice(adjacent_boxes)
                    mergeable = self.env.find_mergeable_boxes(box_idx)
                    if mergeable:
                        target_idx, _ = mergeable[0]
                        return (box_idx, "merge", target_idx)
                        
                # 如果所选类型无可用动作,随机选择一个box和动作
                box_idx = random.randint(0, len(self.env.boxes) - 1)
                return (box_idx, random.choice(self.actions), None)
        else:
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                action_idx = q_values.argmax().item()
                
                if action_idx < len(self.edge_actions):
                    # Edge action
                    action_name = self.edge_actions[action_idx]
                    # 获取所有box的类型信息
                    extreme_boxes = []
                    for box_idx in range(len(self.env.boxes)):
                        is_extreme, _, free_edges = self.get_box_type(box_idx)
                        if is_extreme:
                            extreme_boxes.append((box_idx, free_edges))
                    
                    # 根据动作类型选择合适的box
                    selected_box_idx = 0
                    if extreme_boxes:
                        if action_name in ["shrink_up", "expand_up"]:
                            # 选择y坐标最大的box
                            selected_box_idx = max(range(len(self.env.boxes)), 
                                key=lambda i: self.env.boxes[i]['max_y'])
                        elif action_name in ["shrink_down", "expand_down"]:
                            # 选择y坐标最小的box
                            selected_box_idx = min(range(len(self.env.boxes)), 
                                key=lambda i: self.env.boxes[i]['min_y'])
                        elif action_name in ["shrink_left", "expand_left"]:
                            # 选择x坐标最小的box
                            selected_box_idx = min(range(len(self.env.boxes)), 
                                key=lambda i: self.env.boxes[i]['min_x'])
                        elif action_name in ["shrink_right", "expand_right"]:
                            # 选择x坐标最大的box
                            selected_box_idx = max(range(len(self.env.boxes)), 
                                key=lambda i: self.env.boxes[i]['max_x'])
                    
                    action = (selected_box_idx, action_name, None)
                else:
                    # Merge action - 保持原有的合并逻辑
                    adjacent_boxes = []
                    # 分类所有box
                    for box_idx in range(len(self.env.boxes)):
                        is_extreme, all_adjacent, free_edges = self.get_box_type(box_idx)
                        if all_adjacent:
                            adjacent_boxes.append(box_idx)
                    print(f"adjacent_boxes: {len(self.env.boxes)}")
                    if adjacent_boxes:
                        box_idx = random.choice(adjacent_boxes)
                        mergeable = self.env.find_mergeable_boxes(box_idx)
                        if mergeable :
                            target_idx, _ = mergeable[0]
                            return (box_idx, "merge", target_idx)
                # 如果所选类型无可用动作,随机选择一个box和动作
                box_idx = random.randint(0, len(self.env.boxes) - 1)
                return (box_idx, random.choice(self.actions), None)    
        return action

    def get_box_type(self, box_idx):
        """检查box是否为边界box和四面邻接情况
        
        Args:
            box_idx (int): box的索引

        Returns:
            tuple: (is_extreme, all_adjacent, free_edges)
                - is_extreme (bool): 是否是边界box
                - all_adjacent (bool): 是否四面均有邻接
                - free_edges (dict): 每个边的空闲状态
        """
        box = self.env.boxes[box_idx]
        
        # 判断是否是边界box(是否具有极值坐标)
        min_x = min(b['min_x'] for b in self.env.boxes)
        max_x = max(b['max_x'] for b in self.env.boxes)
        min_y = min(b['min_y'] for b in self.env.boxes)
        max_y = max(b['max_y'] for b in self.env.boxes)
        
        is_extreme = (box['min_x'] == min_x or box['max_x'] == max_x or 
                     box['min_y'] == min_y or box['max_y'] == max_y)
        
        # 检查四个方向的邻接情况
        free_edges = self.env.check_box_adjacency(box_idx)
        all_adjacent = not any(free_edges.values())  # 四面均有邻接
        
        return is_extreme, all_adjacent, free_edges

    def optimize(self):
        if len(self.memory) < self.batch_size:
            return
            
        transitions = random.sample(self.memory, self.batch_size)
        
        state_batch = torch.stack([t[0] for t in transitions])
        action_batch = torch.zeros((self.batch_size, len(self.actions)), device=self.device)
        for i, transition in enumerate(transitions):
            action_idx = self.actions.index(transition[1][1])  # 获取动作在actions列表中的索引
            action_batch[i][action_idx] = 1.0
        
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