import torch
import numpy as np
import cv2
import networkx as nx
import random
from collections import defaultdict, deque
from config import *

def calculate_center_points(indices, size):
    """Calculate the center points based on indices for a given size."""
    center_points = []
    
    # Convert indices to numpy array depending on input type
    if hasattr(indices, 'cpu'):  # Check if indices is a torch tensor
        indices = indices.cpu().numpy()
    elif isinstance(indices, list):
        indices = np.array(indices)
    else:
        indices = np.asarray(indices)

    for index in indices:
        row = index // (size // 14)
        col = index % (size // 14)
        center_x = col * 14 + 14 // 2
        center_y = row * 14 + 14 // 2
        center_points.append([center_x, center_y])

    return center_points

def normalize_distances(distances):
    """Normalize the distances to be between 0 and 1."""
    max_distance = torch.max(distances)
    min_distance = torch.min(distances)
    normalized_distances = (distances - min_distance) / (max_distance - min_distance)
    return normalized_distances

def refine_mask(mask,threshold):

    # Find contours in the mask image
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find the largest contour
    largest_contour = contours[0]

    # Calculate the minimum contour area that is 20% of the size of the largest contour
    min_area = threshold * cv2.contourArea(largest_contour)

    # Find contours that are at least 20% of the size of the largest contour
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]

    # Draw the contours on the resized mask image
    contour_mask = np.zeros_like(mask)
    cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
    cv2.drawContours(contour_mask, filtered_contours, -1, 255, -1)

    return contour_mask

def calculate_distances(features, positive_indices, negative_indices, image_size, device):
    """Calculate feature and physical distances."""
    positive_points = torch.tensor(calculate_center_points(positive_indices, image_size), dtype=torch.float).to(device)
    negative_points = torch.tensor(calculate_center_points(negative_indices, image_size), dtype=torch.float).to(device)

    features = features.to(device)

    feature_positive_distances = torch.cdist(features[1][positive_indices], features[1][positive_indices])
    feature_cross_distances = torch.cdist(features[1][positive_indices], features[1][negative_indices])

    physical_positive_distances = torch.cdist(positive_points, positive_points)
    physical_negative_distances = torch.cdist(negative_points, negative_points)
    physical_cross_distances = torch.cdist(positive_points, negative_points)

    feature_positive_distances = normalize_distances(feature_positive_distances)
    feature_cross_distances = normalize_distances(feature_cross_distances)
    physical_positive_distances = normalize_distances(physical_positive_distances)
    physical_negative_distances = normalize_distances(physical_negative_distances)
    physical_cross_distances = normalize_distances(physical_cross_distances)

    return feature_positive_distances, feature_cross_distances, physical_positive_distances, physical_negative_distances, physical_cross_distances

def draw_points_on_image(image, points, color, size):
    """Draw points on the image."""
    image = np.array(image)
    for point in points:
        cv2.circle(image, (point[0], point[1]), radius=size, color=color, thickness=-1)
    return image

def convert_to_edges(start_nodes, end_nodes, weights):
    """Convert nodes to edges with weights."""
    assert weights.shape == (len(start_nodes), len(end_nodes)), "Weight matrix shape mismatch"
    start_nodes_expanded = start_nodes.unsqueeze(1).expand(-1, end_nodes.size(0))
    end_nodes_expanded = end_nodes.unsqueeze(0).expand(start_nodes.size(0), -1)
    edges_with_weights_tensor = torch.stack((start_nodes_expanded, end_nodes_expanded, weights), dim=2)
    edges_with_weights = edges_with_weights_tensor.view(-1, 3).tolist()
    return edges_with_weights

def average_edge_size(graph, weight_name):
    """Calculate the average edge size based on the specified weight."""
    edges = graph.edges(data=True)
    total_weight = sum(data[weight_name] for _, _, data in edges if weight_name in data)
    edge_count = sum(1 for _, _, data in edges if weight_name in data)
    if edge_count == 0:
        return 0
    average_weight = total_weight / edge_count
    return average_weight

class NodeOptimizationEnv:
    def __init__(self, G, max_steps):
        """Initialize the graph optimization environment."""
        self.original_G = G.copy()
        self.G = G.copy()
        self.pos_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'pos']
        self.neg_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'neg']
        self.min_nodes = 5
        self.max_nodes = 20
        self.steps = 0
        self.max_steps = max_steps
        self.removed_nodes = set()
        self.boxes = []  # 添加boxes属性
        self.reset()

        self.previous_feature_pos_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'feature_pos').values()))
        self.previous_feature_cross_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'feature_cross').values()))
        self.previous_physical_pos_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_pos').values()))
        self.previous_physical_neg_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_neg').values()))
        self.previous_physical_cross_mean = np.mean(list(nx.get_edge_attributes(self.original_G, 'physical_cross').values()))

        self.previous_pos_num = 0
        self.previous_neg_num = 0

    def reset(self):
        """Reset the environment."""
        self.G = self.original_G.copy()
        self.removed_nodes = set(self.G.nodes())
        self.G.clear()
        self.pos_nodes = []
        self.neg_nodes = []
        return self.get_state()

    def get_state(self):
        """Get the current state of the environment."""
        return self.G

    def step(self, action):
        """Perform an action in the environment."""
        node, operation = action
        if operation == "remove_pos":
            self.remove_node(node, "pos")
        elif operation == "remove_neg":
            self.remove_node(node, "neg")
        elif operation == "restore_pos":
            self.restore_node(node, "pos")
        elif operation == "restore_neg":
            self.restore_node(node, "neg")
        elif operation == "add":
            self.add_node(node)

        reward = self.calculate_reward(operation)
        
        if self.min_nodes < len(self.pos_nodes) < self.max_nodes and self.min_nodes < len(self.neg_nodes) < self.max_nodes:
            # 如果reward为负，回滚操作
            if reward < 0:
                self.revertStep(action)
                reward = self.calculate_reward(operation)
        
        self.steps += 1
        done = self.is_done()
        return self.get_state(), reward, done

    def revertStep(self, action):
        node, operation = action
        if operation == "remove_pos":
            self.restore_node(node, "pos")
        elif operation == "remove_neg":
            self.restore_node(node, "neg")
        elif operation == "restore_pos":
            self.remove_node(node, "pos")
        elif operation == "restore_neg":
            self.remove_node(node, "neg")

    def remove_node(self, node, category):
        """Remove a node from the graph."""
        if node in self.G.nodes() and self.G.nodes[node]['category'] == category:
            self.G.remove_node(node)
            self.removed_nodes.add(node)
            if node in self.pos_nodes:
                self.pos_nodes.remove(node)
            elif node in self.neg_nodes:
                self.neg_nodes.remove(node)

    def restore_node(self, node, category):
        """Restore a node to the graph."""
        if node in self.removed_nodes and self.original_G.nodes[node]['category'] == category:
            self.G.add_node(node, **self.original_G.nodes[node])
            self.removed_nodes.remove(node)
            if self.original_G.nodes[node]['category'] == 'pos':
                self.pos_nodes.append(node)
            elif self.original_G.nodes[node]['category'] == 'neg':
                self.neg_nodes.append(node)

            # Restore edges associated with this node
            for neighbor in self.original_G.neighbors(node):
                if neighbor in self.G.nodes():
                    for edge in self.original_G.edges(node, data=True):
                        if edge[1] == neighbor:
                            self.G.add_edge(edge[0], edge[1], **edge[2])

    def add_node(self, node):
        """Add a new node to the graph."""
        category = 'pos' if random.random() < 0.5 else 'neg'
        self.G.add_node(node, category=category)
        self.original_G.add_node(node, category=category)
        if category == 'pos':
            self.pos_nodes.append(node)
        else:
            self.neg_nodes.append(node)

    def calculate_reward(self, operation):
        """Calculate the reward based on the current state and operation."""
        feature_pos_distances = nx.get_edge_attributes(self.G, 'feature_pos')
        feature_cross_distances = nx.get_edge_attributes(self.G, 'feature_cross')
        physical_pos_distances = nx.get_edge_attributes(self.G, 'physical_pos')
        physical_neg_distances = nx.get_edge_attributes(self.G, 'physical_neg')
        physical_cross_distances = nx.get_edge_attributes(self.G, 'physical_cross')

        mean_feature_pos = np.mean(list(feature_pos_distances.values())) if feature_pos_distances else 0
        mean_feature_cross = np.mean(list(feature_cross_distances.values())) if feature_cross_distances else 0
        mean_physical_pos = np.mean(list(physical_pos_distances.values())) if physical_pos_distances else 0
        mean_physical_neg = np.mean(list(physical_neg_distances.values())) if physical_neg_distances else 0
        mean_physical_cross = np.mean(list(physical_cross_distances.values())) if physical_cross_distances else 0

        reward = 0

        if mean_feature_pos < self.previous_feature_pos_mean:
            reward += 3 * (self.previous_feature_pos_mean - mean_feature_pos)
        else:
            reward -= 3 * (mean_feature_pos - self.previous_feature_pos_mean)

        if mean_feature_cross > self.previous_feature_cross_mean:
            reward += 3 * (mean_feature_cross - self.previous_feature_cross_mean)
        else:
            reward -= 3 * (self.previous_feature_cross_mean - mean_feature_cross)

        if mean_physical_pos > self.previous_physical_pos_mean:
            reward += 3 * (mean_physical_pos - self.previous_physical_pos_mean)
        else:
            reward -= 3 *(self.previous_physical_pos_mean - mean_physical_pos)

        if mean_physical_neg > self.previous_physical_neg_mean:
            reward += 1 * (mean_physical_neg - self.previous_physical_neg_mean)
        else:
            reward -= 1 *(self.previous_physical_neg_mean - mean_physical_neg)

        if mean_physical_cross < self.previous_physical_cross_mean:
            reward += 1 * (self.previous_physical_cross_mean - mean_physical_cross)
        else:
            reward -= 1 * (mean_physical_cross - self.previous_physical_cross_mean)

        if operation == "add":
            # 根据add操作后的状态变化调整惩罚
            if (mean_feature_pos < self.previous_feature_pos_mean and 
                mean_feature_cross > self.previous_feature_cross_mean):
                # add操作改善了特征距离，给予较小惩罚
                reward -= 3
            else:
                # add操作没有改善特征距离，给予较大惩罚
                reward -= 5

        self.previous_pos_num = len(self.pos_nodes)
        self.previous_neg_num = len(self.neg_nodes)
        self.previous_feature_cross_mean = mean_feature_cross
        self.previous_feature_pos_mean = mean_feature_pos
        self.previous_physical_pos_mean = mean_physical_pos
        self.previous_physical_neg_mean = mean_physical_neg
        self.previous_physical_cross_mean = mean_physical_cross

        return reward

    def is_done(self):
        """Check if the maximum steps have been reached."""
        return self.steps >= self.max_steps

class NodeAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64,reward_threshold=0.1):
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
        """将图状态转换为哈希值
        
        Args:
            state: 图状态
            
        Returns:
            状态的哈希值
        """
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

        # 只在范围之间添加 add 动作
        if self.env.min_nodes < pos_nodes_count < self.env.max_nodes and self.env.min_nodes < neg_nodes_count < self.env.max_nodes:
            actions.extend(restore_pos_actions)
            actions.extend(restore_neg_actions)
            actions.extend(remove_pos_actions)
            actions.extend(remove_neg_actions)
            # actions.extend([(node, "add") for node in range(state.number_of_nodes(), state.number_of_nodes() + 10)])
        else:
            # 超出上下限范围时，只允许 restore 和 remove 操作
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

def show_mask(mask,ax, random_color=False):

    color = np.array([50/255, 120/255, 255/255, 0.8])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def is_point_in_box(point, bbox):
    """
    判断点是否在边界框内
    
    Args:
        point (list): [x, y] 格式的点坐标
        bbox (dict): 包含 min_x, min_y, max_x, max_y 的边界框信息
    
    Returns:
        bool: 点是否在边界框内
    """
    x, y = point
    return (bbox['min_x'] <= x <= bbox['max_x'] and 
            bbox['min_y'] <= y <= bbox['max_y'])

def is_point_in_any_box(point, boxes):
    """
    判断点是否在任意一个边界框内
    
    Args:
        point (list): [x, y] 格式的点坐标
        boxes (list): box字典列表，每个字典包含min_x,min_y,max_x,max_y
    
    Returns:
        tuple: (bool, int) - 是否在box内及box索引
    """
    x, y = point
    for i, box in enumerate(boxes):
        if (box['min_x'] <= x <= box['max_x'] and 
            box['min_y'] <= y <= box['max_y']):
            return True, i
    return False, -1

def get_box_node_indices(G, boxes):
    """
    获取在边界框内和外的节点索引
    
    Args:
        G (networkx.Graph): 图结构
        boxes (list): box字典列表
    
    Returns:
        tuple: (inside_indices, outside_indices, 
                inside_pos_indices, outside_pos_indices,
                inside_neg_indices, outside_neg_indices)
    """
    inside_indices = []
    outside_indices = []
    inside_pos_indices = []
    outside_pos_indices = []
    inside_neg_indices = []
    outside_neg_indices = []
    
    for node in G.nodes():
        # 计算节点的中心点坐标
        point = calculate_center_points([node], 560)[0]
        
        # 判断点是否在任意box内
        is_inside, _ = is_point_in_any_box(point, boxes)
        
        if is_inside:
            inside_indices.append(node)
            if G.nodes[node]['category'] == 'pos':
                inside_pos_indices.append(node)
            else:
                inside_neg_indices.append(node)
        else:
            outside_indices.append(node)
            if G.nodes[node]['category'] == 'pos':
                outside_pos_indices.append(node)
            else:
                outside_neg_indices.append(node)
    
    return (inside_indices, outside_indices,
            inside_pos_indices, outside_pos_indices,
            inside_neg_indices, outside_neg_indices)

def normalize_distance(distance, prev_distance):
    """归一化距离差值到[-1,1]范围"""
    if prev_distance == 0:
        return 0
    normalized = (prev_distance - distance) / max(abs(prev_distance), abs(distance))
    return max(min(normalized, 1), -1)

class BoxOptimizationEnv:
    def __init__(self, G, bbox_initial=None, image_size=560, max_steps=100, step_size=14):
        """初始化多box优化环境
        
        Args:
            G (nx.Graph): 图结构
            bbox_initial (list): 初始box列表，每个元素为dict包含min_x,min_y,max_x,max_y
            image_size (int): 图像大小
            max_steps (int): 最大步数
            step_size (int): 移动步长
        """
        self.original_G = G.copy()
        self.G = G.copy()
        self.image_size = image_size
        self.step_size = step_size
        self.fine_step_size = self.step_size  # 修改移动步长为14
        self.similarity_threshold = 0.8  # 相似度阈值，用于判断是否合并box
        self.min_box_size = 56  # 最小box尺寸，防止box过小
        self.max_steps = max_steps
        self.bbox_initial = bbox_initial if bbox_initial else []  # 存储初始box配置
        self.boxes = self.bbox_initial.copy()  # 当前box配置
        self.steps = 0
        self.features = None  # 初始化特征
        self.final_box = None  # 添加最终合并box属性
        
        self.previous_inner_coherence = 0  # 记录上一步小box内部一致性
        self.previous_outer_difference = 0  # 记录上一步大box区分性
    
    def reset(self):
        """重置环境"""
        self.G = self.original_G.copy()
        self.boxes = [box.copy() for box in self.bbox_initial]  # 深拷贝初始box配置
        self.steps = 0
        
        self.previous_inner_coherence = 0
        self.previous_outer_difference = 0
        return self.get_state()

    def get_state(self):
        """获取当前状态"""
        return (self.G, self.boxes)

    def set_features(self, features):
        """设置特征"""
        self.features = features

    def calculate_similarity(self, features):
        """计算特征之间的余弦相似度
        
        Args:
            features (torch.Tensor): 特征向量张量
            
        Returns:
            float: 平均相似度
        """
        if features is None or features.shape[0] < 2:
            return 0
            
        # 特征标准化
        normalized_features = torch.nn.functional.normalize(features, p=2, dim=1)
        
        # 计算余弦相似度
        similarities = torch.mm(normalized_features, normalized_features.t())
        
        # 移除自身相似度
        mask = torch.eye(similarities.shape[0], device=similarities.device)
        similarities = similarities * (1 - mask)
        
        # 计算平均相似度
        mean_similarity = similarities.sum() / (similarities.shape[0] * (similarities.shape[0] - 1))
        
        return mean_similarity.item()

    def get_box_features(self, box_idx):
        """获取指定box内的特征
        基于patch位置计算box内的特征,而不是基于节点位置
        """
        if self.features is None:
            return None
        
        box = self.boxes[box_idx]
        patch_size = 14  # DINO特征的patch大小
        
        # 计算box覆盖的patch范围
        start_row = box['min_y'] // patch_size
        end_row = (box['max_y'] + patch_size - 1) // patch_size
        start_col = box['min_x'] // patch_size
        end_col = (box['max_x'] + patch_size - 1) // patch_size
        
        # 获取所有在box内的patch索引
        patch_indices = []
        for row in range(start_row, end_row):
            for col in range(start_col, end_col):
                # 计算patch的中心点
                center_x = col * patch_size + patch_size // 2
                center_y = row * patch_size + patch_size // 2
                
                # 检查patch中心点是否在box内
                if (box['min_x'] <= center_x <= box['max_x'] and 
                    box['min_y'] <= center_y <= box['max_y']):
                    # 将2D索引转换为1D索引
                    index = row * (self.image_size // patch_size) + col
                    patch_indices.append(index)
        
        # 如果没有找到有效的patch,返回None
        if not patch_indices:
            return None
            
        # 获取patch的特征
        patch_features = self.features[1][patch_indices]
        
        # 在函数末尾添加特征标准化
        if patch_features is not None:
            # 标准化特征
            patch_features = torch.nn.functional.normalize(patch_features, p=2, dim=1)
        return patch_features

    def calculate_inner_coherence(self):
        """计算所有小box的内部特征一致性平均值"""
        if not self.boxes:
            return 0
            
        coherence_values = []
        for box_idx in range(len(self.boxes)):
            box_features = self.get_box_features(box_idx)
            if box_features is not None and len(box_features) > 1:
                # 计算box内特征的余弦相似度
                similarity = self.calculate_similarity(box_features)
                coherence_values.append(similarity)
                
        return np.mean(coherence_values) if coherence_values else 0
    
    def calculate_outer_difference(self):
        """计算合并后大box的内外特征区分性"""
        # 构建大box
        final_box = {
            'min_x': min(box['min_x'] for box in self.boxes),
            'min_y': min(box['min_y'] for box in self.boxes),
            'max_x': max(box['max_x'] for box in self.boxes),
            'max_y': max(box['max_y'] for box in self.boxes)
        }
        
        if self.features is None:
            return 0
            
        patch_size = 14
        # 获取大box内的所有特征
        inner_features = []
        outer_features = []
        
        for idx in range(self.features[1].shape[0]):
            # 计算patch中心点
            row = idx // (self.image_size // patch_size)
            col = idx % (self.image_size // patch_size)
            center_x = col * patch_size + patch_size // 2
            center_y = row * patch_size + patch_size // 2
            
            # 判断patch是否在大box内
            if (final_box['min_x'] <= center_x <= final_box['max_x'] and 
                final_box['min_y'] <= center_y <= final_box['max_y']):
                inner_features.append(self.features[1][idx])
            else:
                outer_features.append(self.features[1][idx])
                
        if not inner_features or not outer_features:
            return 0
            
        inner_features = torch.stack(inner_features)
        outer_features = torch.stack(outer_features)
        
        # 计算内外特征的平均余弦距离
        inner_mean = torch.mean(inner_features, dim=0, keepdim=True)
        outer_mean = torch.mean(outer_features, dim=0, keepdim=True)
        
        difference = 1 - torch.nn.functional.cosine_similarity(inner_mean, outer_mean)
        return difference.item()

    def calculate_box_reward(self, box_idx):
        """box的奖励计算,基于内部一致性和外部区分性"""
        # 计算当前状态的内部一致性和外部区分性
        current_coherence = self.calculate_inner_coherence()
        current_difference = self.calculate_outer_difference()
        
        # 计算变化量
        coherence_change = normalize_distance(
            current_coherence, 
            self.previous_inner_coherence)
        difference_change = normalize_distance(
            current_difference,
            self.previous_outer_difference)
            
        # 更新历史值
        self.previous_inner_coherence = current_coherence
        self.previous_outer_difference = current_difference
        
        # 计算总奖励
        # 内部一致性提高(positive change)给予正奖励
        # 外部区分性提高(positive change)给予正奖励
        reward = 1.0 * coherence_change + 2.0 * difference_change
        
        return reward

    def check_box_adjacency(self, box_idx):
        """检查box是否有空闲边（没有邻接其他box的边）"""
        box = self.boxes[box_idx]
        edges = {
            'up': False, 'down': False,
            'left': False, 'right': False
        }
        
        for i, other_box in enumerate(self.boxes):
            if i == box_idx:
                continue
                
            # 检查上边
            if (abs(box['min_y'] - other_box['max_y']) <= self.step_size and
                max(box['min_x'], other_box['min_x']) < min(box['max_x'], other_box['max_x'])):
                edges['up'] = True
                
            # 检查下边
            if (abs(box['max_y'] - other_box['min_y']) <= self.step_size and
                max(box['min_x'], other_box['min_x']) < min(box['max_x'], other_box['max_x'])):
                edges['down'] = True
                
            # 检查左边
            if (abs(box['min_x'] - other_box['max_x']) <= self.step_size and
                max(box['min_y'], other_box['min_y']) < min(box['max_y'], other_box['max_y'])):
                edges['left'] = True
                
            # 检查右边
            if (abs(box['max_x'] - other_box['min_x']) <= self.step_size and
                max(box['min_y'], other_box['min_y']) < min(box['max_y'], other_box['max_y'])):
                edges['right'] = True
                
        return {k: not v for k, v in edges.items()}  # 返回空闲边

    def find_mergeable_boxes(self, box_idx):
        """查找可以合并的相邻box"""
        box = self.boxes[box_idx]
        mergeable = []
        
        for i, other_box in enumerate(self.boxes):
            if i == box_idx:
                continue
                
            # 检查是否相邻
            is_adjacent = (
                (abs(box['min_y'] - other_box['max_y']) <= self.step_size or
                 abs(box['max_y'] - other_box['min_y']) <= self.step_size) and
                max(box['min_x'], other_box['min_x']) < min(box['max_x'], other_box['max_x'])
            ) or (
                (abs(box['min_x'] - other_box['max_x']) <= self.step_size or
                 abs(box['max_x'] - other_box['min_x']) <= self.step_size) and
                max(box['min_y'], other_box['min_y']) < min(box['max_y'], other_box['max_y'])
            )
            
            if is_adjacent:
                # 计算特征相似度
                box_features = self.get_box_features(box_idx)
                other_features = self.get_box_features(i)
                if box_features is not None and other_features is not None:
                    similarity = self.calculate_box_similarity(box_features, other_features)
                    if similarity > self.similarity_threshold:
                        mergeable.append((i, similarity))
        
        return sorted(mergeable, key=lambda x: x[1], reverse=True)

    def calculate_box_similarity(self, features1, features2):
        """计算两个box的特征相似度"""
        if features1 is None or features2 is None:
            return 0
            
        # 计算特征的平均值
        mean_feature1 = torch.mean(features1, dim=0, keepdim=True)
        mean_feature2 = torch.mean(features2, dim=0, keepdim=True)
        
        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(mean_feature1, mean_feature2)
        return similarity.item()

    def merge_boxes(self, box1_idx, box2_idx):
        """合并两个box"""
        box1 = self.boxes[box1_idx]
        box2 = self.boxes[box2_idx]
        
        # 创建新的合并box
        merged_box = {
            'min_x': min(box1['min_x'], box2['min_x']),
            'min_y': min(box1['min_y'], box2['min_y']),
            'max_x': max(box1['max_x'], box2['max_x']),
            'max_y': max(box1['max_y'], box2['max_y'])
        }
        
        # 移除旧box并添加新box
        self.boxes.pop(max(box1_idx, box2_idx))
        self.boxes.pop(min(box1_idx, box2_idx))
        self.boxes.append(merged_box)
        
        return len(self.boxes) - 1  # 返回新box的索引

    def step(self, action):
        box_idx, operation, params = action
        if box_idx >= len(self.boxes):
            return self.get_state(), 0, True
            
        old_box = self.boxes[box_idx].copy()
        old_boxes = self.boxes.copy()
        
        reward = 0
        if operation == "merge" and params is not None:
            # 执行合并操作
            target_idx = params
            new_box_idx = self.merge_boxes(box_idx, target_idx)
            reward = self.calculate_box_reward(new_box_idx)
        else:
            free_edges = self.check_box_adjacency(box_idx)
            if operation.startswith(("shrink_", "expand_")):
                direction = operation.split("_")[1]
                if free_edges[direction]:
                    amount = self.step_size if operation.startswith("expand") else -self.step_size
                    # 根据方向调整box大小
                    if direction == "up":
                        new_val = max(0, self.boxes[box_idx]['min_y'] + amount)
                        if new_val < self.boxes[box_idx]['max_y'] - self.min_box_size:
                            self.boxes[box_idx]['min_y'] = new_val
                    elif direction == "down":
                        new_val = min(self.image_size, self.boxes[box_idx]['max_y'] + amount)
                        if new_val > self.boxes[box_idx]['min_y'] + self.min_box_size:
                            self.boxes[box_idx]['max_y'] = new_val
                    elif direction == "left":
                        new_val = max(0, self.boxes[box_idx]['min_x'] + amount)
                        if new_val < self.boxes[box_idx]['max_x'] - self.min_box_size:
                            self.boxes[box_idx]['min_x'] = new_val
                    elif direction == "right":
                        new_val = min(self.image_size, self.boxes[box_idx]['max_x'] + amount)
                        if new_val > self.boxes[box_idx]['min_x'] + self.min_box_size:
                            self.boxes[box_idx]['max_x'] = new_val
                    
                    reward = self.calculate_box_reward(box_idx)
        
        print(f"box_reward{reward}")
        # 如果奖励为负，回滚操作
        if reward < 0:
            self.boxes = old_boxes
            reward = 0
        else:
            self.steps += 1
        
        done = self.steps >= self.max_steps
        return self.get_state(), reward, done

    def calculate_pos_ratio_in_box(self):
        """计算所有box内正样本比例"""
        pos_nodes = [n for n, d in self.G.nodes(data=True) if d['category'] == 'pos']
        if not pos_nodes:
            return 0
        
        inside_count = 0
        for node in pos_nodes:
            point = calculate_center_points([node], self.image_size)[0]
            is_inside, _ = is_point_in_any_box(point, self.boxes)
            if is_inside:
                inside_count += 1
        
        return inside_count / len(pos_nodes)
    
    def calculate_neg_ratio_out_box(self):
        """计算所有box外负样本比例"""
        neg_nodes = [n for n, d in self.G.nodes(data=True) if d['category'] == 'neg']
        if not neg_nodes:
            return 0
        
        outside_count = 0
        for node in neg_nodes:
            point = calculate_center_points([node], self.image_size)[0]
            is_inside, _ = is_point_in_any_box(point, self.boxes)
            if not is_inside:
                outside_count += 1
        
        return outside_count / len(neg_nodes)

class BoxAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        """初始化box智能体"""
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start  # 存储初始epsilon值
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.q_table = defaultdict(float)
        
        # 添加其他必要的属性
        self.best_reward = -float('inf')  # 用于记录最佳奖励
        self.best_reward_save = -float('inf')
        self.memory = deque(maxlen=10000)  # 添加经验回放内存
        self.best_memory = deque(maxlen=10000)  # 添加最佳经验内存
        self.batch_size = 64  # 添加批量大小
        
        # 动作空间设计
        self.edge_actions = ["shrink_up", "shrink_down", "shrink_left", "shrink_right",
                             "expand_up", "expand_down", "expand_left", "expand_right"]
        self.merge_action = ["merge"]
        self.actions = self.edge_actions + self.merge_action

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_box_type(self, box_idx):
        """判断box类型:是否为边界box,是否四面均有邻接"""
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

    def get_action(self, state):
        """根据box类型选择合适的动作"""
        if random.random() < self.epsilon:
            # 50%概率选择极值box,50%概率选择四面邻接box
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
            # 使用Q表选择动作
            state_hash = self._get_state_hash(state)
            best_value = float('-inf')
            best_action = None
            
            # 同样遵循50%的概率分配
            if random.random() < 0.5:
                # 评估极值box的动作
                for box_idx in range(len(self.env.boxes)):
                    is_extreme, _, free_edges = self.get_box_type(box_idx)
                    if is_extreme:
                        for direction, is_free in free_edges.items():
                            if is_free:
                                for op in [f"shrink_{direction}", f"expand_{direction}"]:
                                    action = (box_idx, op, None)
                                    value = self.q_table.get((state_hash, action), 0)
                                    if value > best_value:
                                        best_value = value
                                        best_action = action
            else:
                # 评估四面邻接box的合并动作
                for box_idx in range(len(self.env.boxes)):
                    _, all_adjacent, _ = self.get_box_type(box_idx)
                    if all_adjacent:
                        mergeable = self.env.find_mergeable_boxes(box_idx)
                        for target_idx, similarity in mergeable:
                            action = (box_idx, "merge", target_idx)
                            # 在Q值基础上加入相似度作为额外考虑因素
                            value = self.q_table.get((state_hash, action), 0) + similarity
                            if value > best_value:
                                best_value = value
                                best_action = action
            
            return best_action if best_action else (0, "shrink_up", None)

    def _get_state_hash(self, state):
        """将状态转换为哈希值"""
        _, boxes = state
        box_tuples = tuple(
            (box['min_x'], box['min_y'], box['max_x'], box['max_y'])
            for box in boxes
        )
        return hash(box_tuples)

    def update_q_table(self, state, action, reward, next_state):
        """更新Q表,根据box类型选择动作空间"""
        state_hash = self._get_state_hash(state)
        next_state_hash = self._get_state_hash(next_state)
        
        # 获取next_state中每个box可用的动作
        next_actions = []
        for i in range(len(self.env.boxes)):
            is_extreme, all_adjacent, free_edges = self.get_box_type(i)
            if is_extreme:
                # 边界box的expand/shrink动作
                for direction, is_free in free_edges.items():
                    if is_free:
                        next_actions.extend([
                            (i, f"shrink_{direction}", None),
                            (i, f"expand_{direction}", None)
                        ])
            elif all_adjacent:
                # 四面邻接box的合并动作
                mergeable = self.env.find_mergeable_boxes(i)
                for target_idx, _ in mergeable:
                    next_actions.append((i, "merge", target_idx))
        
        # 计算最大Q值
        max_next_q = max(
            [self.q_table.get((next_state_hash, next_action), 0) 
             for next_action in next_actions],
            default=0)
        
        old_q = self.q_table.get((state_hash, action), 0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state_hash, action)] = new_q

    def replay(self):
        """从记忆中回放经验"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)

    def replay_best(self):
        """从最佳记忆中回放经验"""
        if len(self.best_memory) < self.batch_size:
            return
        
        batch = random.sample(self.best_memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)

class MultiAgentEnv:
    def __init__(self, G, bbox_initial=None, image_size=560, max_steps=100, step_size=14):
        """初始化多智能体环境

        Args:
            G (nx.Graph): 图结构
            bbox_initial (list): 初始边界框 [min_x, min_y, max_x, max_y]
            image_size (int): 图像大小
            max_steps (int): 最大步数
            step_size (int): 移动步长
        """
        self.node_env = NodeOptimizationEnv(G, max_steps)
        self.box_env = BoxOptimizationEnv(G, bbox_initial, image_size, max_steps, step_size)
        self.node_env.boxes = self.box_env.boxes  # 初始化时共享boxes
        self.steps = 0
        self.max_steps = max_steps
        self.features = None  # 添加特征存储
        
    def reset(self):
        """重置环境"""
        node_state = self.node_env.reset()
        box_state = self.box_env.reset()
        self.steps = 0
        
        # 确保特征在重置时也被正确设置
        if hasattr(self.node_env, 'features'):
            self.features = self.node_env.features
            self.box_env.features = self.features
            
        return {"node": node_state, "box": box_state}
    
    def step(self, action_dict):
        """执行动作并计算综合奖励
        
        Args:
            action_dict: 包含节点和边界框动作的字典
                {"node": node_action, "box": box_action}
        
        Returns:
            next_state: 下一个状态
            reward: 奖励
            done: 是否完成
        """      
        # 执行节点智能体的动作
        if "node" in action_dict:
            node_next_state, node_reward, node_done = self.node_env.step(action_dict["node"])
            # 同步图结构和特征到box环境
            self.box_env.G = self.node_env.G.copy()
            self.node_env.boxes = self.box_env.boxes  # 保持boxes同步
            if hasattr(self.node_env, 'features'):
                self.features = self.node_env.features
                self.box_env.features = self.features
        else:
            node_next_state, node_reward, node_done = self.node_env.get_state(), 0, False
        
        # 执行边界框智能体的动作
        if "box" in action_dict:
            box_next_state, box_reward, box_done = self.box_env.step(action_dict["box"])
        else:
            box_next_state, box_reward, box_done = self.box_env.get_state(), 0, False
        
        # 更新步数
        self.steps += 1
        
        # 组合奖励
        reward = node_reward + box_reward
        done = node_done or box_done or self.steps >= self.max_steps

        # 检查是否达到最大步数
        if self.steps >= self.max_steps:
            # 获取所有box内的负提示点
            neg_nodes_to_remove = []
            for node in self.node_env.G.nodes():
                if self.node_env.G.nodes[node]['category'] == 'neg':
                    point = calculate_center_points([node], self.box_env.image_size)[0]
                    # 检查点是否在任意box内
                    for box in self.box_env.boxes:
                        if (box['min_x'] <= point[0] <= box['max_x'] and 
                            box['min_y'] <= point[1] <= box['max_y']):
                            neg_nodes_to_remove.append(node)
                            break
            
            # 移除这些负提示点
            for node in neg_nodes_to_remove:
                self.node_env.remove_node(node, "neg")
                # 确保box环境的图也保持同步
                if node in self.box_env.G.nodes():
                    self.box_env.G.remove_node(node)
            
            # 在移除节点后更新状态
            node_next_state = self.node_env.get_state()
            box_next_state = self.box_env.get_state()
        
        return {"node": node_next_state, "box": box_next_state}, reward, done
    
    def get_state(self):
        """获取当前状态"""
        return {"node": self.node_env.get_state(), "box": self.box_env.get_state()}
