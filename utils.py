import torch
import numpy as np
from PIL import Image
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

def calculate_bounding_box(points, patch_size=14, image_size=560):
    """
    计算包围所有点的边界框，并进行适当扩展
    
    Parameters:
    points (list): 点的列表，每个点是[x, y]格式
    patch_size (int): 特征patch的大小
    image_size (int): 图像的大小
    
    Returns:
    tuple: (min_x, min_y, max_x, max_y)
    """
    if not points:
        return None
    
    points = np.array(points)
    min_x, min_y = np.min(points, axis=0)
    max_x, max_y = np.max(points, axis=0)
    
    # 扩展边界框，扩展1.5倍patch大小
    padding = int(patch_size * 1.5)
    min_x = max(0, min_x - padding)
    min_y = max(0, min_y - padding)
    max_x = min(image_size, max_x + padding)
    max_y = min(image_size, max_y + padding)
    
    return min_x, min_y, max_x, max_y

def map_to_original_size(resized_coordinates, original_size, image_size):
    """Map resized coordinates back to the original image size."""
    original_height, original_width = original_size
    scale_height = original_height / image_size
    scale_width = original_width / image_size

    if isinstance(resized_coordinates, tuple):
        resized_x, resized_y = resized_coordinates
        original_x = resized_x * scale_width
        original_y = resized_y * scale_height
        return original_x, original_y
    elif isinstance(resized_coordinates, list):
        original_coordinates = [[round(x * scale_width), round(y * scale_height)] for x, y in resized_coordinates]
        return original_coordinates
    else:
        raise ValueError("Unsupported input format. Please provide a tuple or list of coordinates.")

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

def generate_points(positive_indices, negative_indices, image_size):
    """Generate positive and negative points mapped to original size."""
    positive_points = calculate_center_points(positive_indices, image_size)
    negative_points = calculate_center_points(negative_indices, image_size)

    unique_positive_points = set(tuple(point) for point in positive_points)
    unique_negative_points = set(tuple(point) for point in negative_points)

    mapped_positive_points = map_to_original_size(list(unique_positive_points), [560, 560], image_size)
    mapped_negative_points = map_to_original_size(list(unique_negative_points), [560, 560], image_size)

    return mapped_positive_points, mapped_negative_points

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
            if reward < 0:
                # print("node action reward < 0 , revert step")
                self.revertStep(action)
                reward = self.calculate_reward(operation)
            elif reward > 0:
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
            reward += 2 * (self.previous_feature_pos_mean - mean_feature_pos)
        else:
            reward -= 2 * (mean_feature_pos - self.previous_feature_pos_mean)

        if mean_feature_cross > self.previous_feature_cross_mean:
            reward += 2 * (mean_feature_cross - self.previous_feature_cross_mean)
        else:
            reward -= 2 * (self.previous_feature_cross_mean - mean_feature_cross)

        if mean_physical_pos > self.previous_physical_pos_mean:
            reward += (mean_physical_pos - self.previous_physical_pos_mean)
        else:
            reward -= (self.previous_physical_pos_mean - mean_physical_pos)

        if mean_physical_neg > self.previous_physical_neg_mean:
            reward += (mean_physical_neg - self.previous_physical_neg_mean)
        else:
            reward -= (self.previous_physical_neg_mean - mean_physical_neg)

        if mean_physical_cross < self.previous_physical_cross_mean:
            reward += (self.previous_physical_cross_mean - mean_physical_cross)
        else:
            reward -= (mean_physical_cross - self.previous_physical_cross_mean)

        if operation == "add":
            # reward -= 10  # Add penalty for add operation
            ''' 调整add操作的惩罚机制 '''
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

    def get_action(self, state):
        """Select an action based on epsilon-greedy policy."""
        actions = self.get_possible_actions(state)

        if random.random() < self.epsilon:
            action = random.choice(actions)
        else:
            # Select the action with the highest Q-value
            q_values = {action: self.q_table[(state, action)] for action in actions}
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

        # 取消重复的添加删除操作的动作
        # 保证 remove_pos 和 remove_neg 行为的数量一致
        # if len(remove_pos_actions) < len(remove_neg_actions):
        #     remove_neg_actions = random.sample(remove_neg_actions, len(remove_pos_actions))
        # elif len(remove_neg_actions) < len(remove_pos_actions):
        #     remove_pos_actions = random.sample(remove_pos_actions, len(remove_neg_actions))

        # actions.extend(remove_pos_actions)
        # actions.extend(remove_neg_actions)
        
        ''' debug:检测是否存在重复行为 '''
        # print(f"actions length: {len(actions)}")
        # print(f"set actions length: {len(set(actions))}")

        return actions

    def update_q_table(self, state, action, reward, next_state):
        """Update the Q-table based on the current state, action, reward, and next state."""
        max_next_q = max(
            [self.q_table[(next_state, next_action)] for next_action in self.get_possible_actions(next_state)],
            default=0)
        self.q_table[(state, action)] += self.alpha * (reward + self.gamma * max_next_q - self.q_table[(state, action)])

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

def sample_points(mask_path, new_size=(224, 224), num_positive=10, num_negative=10):
    """随机采样正点和负点坐标"""
    mask = Image.open(mask_path).convert("L").resize(new_size)
    mask_array = np.array(mask)

    # 获取白色（正点）和黑色（负点）的坐标
    positive_points = np.column_stack(np.where(mask_array == 255))
    negative_points = np.column_stack(np.where(mask_array == 0))

    # 如果正点或负点的数量不足指定的数量，使用实际数量
    num_positive = min(num_positive, len(positive_points))
    num_negative = min(num_negative, len(negative_points))

    # 随机采样正点和负点
    sampled_positive_points = positive_points[np.random.choice(len(positive_points), num_positive, replace=False)]
    sampled_negative_points = negative_points[np.random.choice(len(negative_points), num_negative, replace=False)]

    positive_indices = calculate_block_index(sampled_positive_points,560)
    negative_indices =calculate_block_index(sampled_negative_points,560)

    return mask_array, positive_indices, negative_indices


def calculate_block_index(center_points, size, block_size=14):
    """根据中心点坐标计算块索引"""
    indices = []
    for (y, x) in center_points:
        row = y // block_size
        col = x // block_size
        index = row * (size // block_size) + col
        indices.append(index)
    return indices

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

def get_box_node_indices(G, bbox):
    """
    获取在边界框内和外的节点索引
    
    Args:
        G (networkx.Graph): 图结构
        bbox (dict): 包含 min_x, min_y, max_x, max_y 的边界框信息
    
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
        
        # 判断点是否在边界框内
        if is_point_in_box(point, bbox):
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
    def __init__(self, G, bbox_initial, image_size=560, max_steps=100, step_size=14):
        """初始化矩形框优化环境。
        
        Args:
            G (nx.Graph): 图结构
            bbox_initial (dict): 包含 min_x, min_y, max_x, max_y 的初始边界框
            image_size (int): 图像大小
            max_steps (int): 最大步数
            step_size (int): 移动步长（像素）
        """
        self.original_G = G.copy()
        self.G = G.copy()
        self.image_size = image_size
        self.step_size = step_size
        self.max_steps = max_steps
        self.bbox_initial = bbox_initial
        self.steps = 0
        self.features = None  # 存储特征
        
        # 初始化边界框，如果没有提供则创建默认边界框
        if bbox_initial is None:
            # 默认边界框为图像中心的1/4大小
            center = image_size // 2
            size = image_size // 4
            self.bbox = {
                'min_x': center - size // 2,
                'min_y': center - size // 2,
                'max_x': center + size // 2,
                'max_y': center + size // 2
            }
        else:
            # 直接使用传入的字典
            self.bbox = bbox_initial
        
        # 获取初始节点索引
        self.update_node_indices()

    def calculate_feature_distances(self):
        """
        计算特征距离，返回框内外节点与groundtruth的距离以及框内外节点间的距离
        
        Returns:
            tuple: (inside_gt_distance, outside_gt_distance, feature_distance)
                  如果无法计算则返回(1.0, 0.0, 0.0)
        """
        # 检查是否有有效的特征和节点
        if (self.features is None or 
            not self.inside_indices or 
            not self.outside_indices):
            return 1.0, 0.0, 0.0

        # 将索引转换为tensor
        inside_indices = torch.tensor(self.inside_indices, dtype=torch.long)
        outside_indices = torch.tensor(self.outside_indices, dtype=torch.long)

        # 获取特征
        inside_features = self.features[1][inside_indices]  # [N, D]
        outside_features = self.features[1][outside_indices]  # [M, D]
        gt_features = self.features[0]  # groundtruth特征

        # 计算平均特征向量
        inside_mean = torch.mean(inside_features, dim=0)  # [D]
        outside_mean = torch.mean(outside_features, dim=0)  # [D]
        gt_mean = torch.mean(gt_features, dim=0)  # [D]

        # 计算距离
        inside_gt_distance = torch.norm(inside_mean - gt_mean).item()
        outside_gt_distance = torch.norm(outside_mean - gt_mean).item()
        feature_distance = torch.norm(inside_mean - outside_mean).item()

        return inside_gt_distance, outside_gt_distance, feature_distance
    
    def update_node_indices(self):
        """更新节点索引"""
        (self.inside_indices, self.outside_indices,
         self.inside_pos_indices, self.outside_pos_indices,
         self.inside_neg_indices, self.outside_neg_indices) = get_box_node_indices(self.G, self.bbox)
    
    def reset(self):
        """重置环境"""
        # 重置图结构
        self.G = self.original_G.copy()
        # 重置边界框到初始状态
        self.bbox = self.bbox_initial
        self.steps = 0
        self.update_node_indices()
        self.previous_pos_ratio_in_box = self.calculate_pos_ratio_in_box()
        self.previous_neg_ratio_out_box = self.calculate_neg_ratio_out_box()
        self.previous_inside_feature_distance, self.previous_outside_feature_distance, self.previous_feature_distance = self.calculate_feature_distances()
        return self.get_state()
    
    def get_state(self):
        """获取当前状态"""
        return (self.G, self.bbox)
    
    def calculate_pos_ratio_in_box(self):
        """计算框内正样本节点比例"""
        pos_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'pos']
        if not pos_nodes:
            return 0
        return len(self.inside_pos_indices) / len(pos_nodes)
    
    def calculate_neg_ratio_out_box(self):
        """计算框外负样本节点比例"""
        neg_nodes = [node for node, data in self.G.nodes(data=True) if data['category'] == 'neg']
        if not neg_nodes:
            return 0
        return len(self.outside_neg_indices) / len(neg_nodes)
    
    def step(self, action):
        """执行动作
        
        动作空间:
        - 'move_up': 向上移动边界框
        - 'move_down': 向下移动边界框
        - 'move_left': 向左移动边界框
        - 'move_right': 向右移动边界框
        - 'increase_width': 增加边界框宽度
        - 'decrease_width': 减少边界框宽度
        - 'increase_height': 增加边界框高度
        - 'decrease_height': 减少边界框高度
        """
        # 保存旧的边界框
        old_bbox = self.bbox.copy()
        
        # 根据动作更新边界框
        if action == "move_up":
            self.bbox['min_y'] = max(0, self.bbox['min_y'] - self.step_size)
            self.bbox['max_y'] = max(self.step_size, self.bbox['max_y'] - self.step_size)
        elif action == "move_down":
            self.bbox['min_y'] = min(self.image_size - self.step_size, self.bbox['min_y'] + self.step_size)
            self.bbox['max_y'] = min(self.image_size, self.bbox['max_y'] + self.step_size)
        elif action == "move_left":
            self.bbox['min_x'] = max(0, self.bbox['min_x'] - self.step_size)
            self.bbox['max_x'] = max(self.step_size, self.bbox['max_x'] - self.step_size)
        elif action == "move_right":
            self.bbox['min_x'] = min(self.image_size - self.step_size, self.bbox['min_x'] + self.step_size)
            self.bbox['max_x'] = min(self.image_size, self.bbox['max_x'] + self.step_size)
        elif action == "increase_width":
            # 增加宽度，左右两边各扩展step_size
            self.bbox['min_x'] = max(0, self.bbox['min_x'] - self.step_size)
            self.bbox['max_x'] = min(self.image_size, self.bbox['max_x'] + self.step_size)
        elif action == "decrease_width":
            # 确保边界框宽度不小于step_size
            if self.bbox['max_x'] - self.bbox['min_x'] > self.step_size*2:
                self.bbox['min_x'] += self.step_size
                self.bbox['max_x'] -= self.step_size
        elif action == "increase_height":
            # 增加高度，上下两边各扩展step_size
            self.bbox['min_y'] = max(0, self.bbox['min_y'] - self.step_size)
            self.bbox['max_y'] = min(self.image_size, self.bbox['max_y'] + self.step_size)
        elif action == "decrease_height":
            # 确保边界框高度不小于step_size
            if self.bbox['max_y'] - self.bbox['min_y'] > self.step_size*2:
                self.bbox['min_y'] += self.step_size
                self.bbox['max_y'] -= self.step_size
        
        print(self.bbox)
        # 更新节点索引
        self.update_node_indices()
        
        # 计算奖励
        reward = self.calculate_reward()
        
        # 如果奖励为负，撤销操作
        if reward < 0:
            # print("box action reward < 0 , revert action")
            self.bbox = old_bbox
            self.update_node_indices()
            reward = self.calculate_reward()
        else:
            self.steps += 1
        
        done = self.is_done()
        return self.get_state(), reward, done
    
    def calculate_reward(self):
        """计算奖励"""
        # 计算当前状态的指标
        current_pos_ratio_in_box = self.calculate_pos_ratio_in_box()
        current_neg_ratio_out_box = self.calculate_neg_ratio_out_box()
        
        # 计算奖励
        reward = 0
        
        # 正样本框内比例奖励
        pos_ratio_diff = current_pos_ratio_in_box - self.previous_pos_ratio_in_box
        reward += 5 * max(min(pos_ratio_diff, 1), -1)

        # 负样本框外比例奖励
        neg_ratio_diff = current_neg_ratio_out_box - self.previous_neg_ratio_out_box
        reward += 1 * max(min(neg_ratio_diff, 1), -1)

        # 计算特征距离奖励
        if self.features is not None and len(self.inside_indices) > 0 and len(self.outside_indices) > 0:           
            # 计算当前特征距离
            inside_gt_distance, outside_gt_distance, inout_feature_distance = self.calculate_feature_distances()

            # 归一化并计算奖励
            inside_distance_norm = normalize_distance(inside_gt_distance, self.previous_inside_feature_distance)
            outside_distance_norm = normalize_distance(outside_gt_distance, self.previous_outside_feature_distance)
            feature_diff_norm = normalize_distance(inout_feature_distance, self.previous_feature_distance)

            # 框内特征与gt距离减小时给予奖励
            reward += 2 * inside_distance_norm

            # 框外特征与gt距离增大时给予奖励
            reward += 2 * outside_distance_norm

            # 框内外特征差异增大时给予奖励
            reward += 2 * feature_diff_norm
            
            # 更新上一状态的指标
            self.previous_pos_ratio_in_box = self.calculate_pos_ratio_in_box()
            self.previous_neg_ratio_out_box = self.calculate_neg_ratio_out_box()
            self.previous_inside_feature_distance = inside_gt_distance
            self.previous_outside_feature_distance = outside_gt_distance
            self.previous_feature_distance = inout_feature_distance

        return reward
    
    def is_done(self):
        """检查是否完成"""
        return self.steps >= self.max_steps

class BoxAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995, memory_size=10000, batch_size=64):
        """初始化矩形框智能体
        
        Args:
            env: 环境
            alpha: 学习率
            gamma: 折扣因子
            epsilon_start: 初始探索率
            epsilon_end: 最终探索率
            epsilon_decay: 探索率衰减
            memory_size: 记忆大小
            batch_size: 批量大小
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(float)
        self.memory = deque(maxlen=memory_size)
        self.best_memory = deque(maxlen=memory_size)
        self.batch_size = batch_size
        self.best_reward = -float('inf')
        self.best_q_table = None
        self.actions = ["move_up", "move_down", "move_left", "move_right", 
                        "increase_width", "decrease_width", 
                        "increase_height", "decrease_height"]
    
    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def get_action(self, state):
        """获取动作
        
        Args:
            state: 环境状态
        
        Returns:
            选择的动作
        """
        # 使用ε-贪婪策略选择动作
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            # 找到具有最高Q值的动作
            state_hash = self._get_state_hash(state)
            q_values = {action: self.q_table.get((state_hash, action), 0) for action in self.actions}
            max_q = max(q_values.values())
            # 如果有多个具有最大Q值的动作，随机选择一个
            best_actions = [action for action, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def _get_state_hash(self, state):
        """将状态转换为哈希值
        
        Args:
            state: 环境状态
        
        Returns:
            状态的哈希值
        """
        _, bbox = state
        # 使用边界框坐标作为状态特征
        bbox_tuple = (
            bbox['min_x'], bbox['min_y'], 
            bbox['max_x'], bbox['max_y']
        )
        return hash(bbox_tuple)
    
    def update_q_table(self, state, action, reward, next_state):
        """更新Q表
        
        Args:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一个状态
        """
        state_hash = self._get_state_hash(state)
        next_state_hash = self._get_state_hash(next_state)
        
        # 计算下一个状态的最大Q值
        max_next_q = max([self.q_table.get((next_state_hash, a), 0) for a in self.actions])
        
        # 更新Q值
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
        self.steps = 0
        self.max_steps = max_steps
        
    def reset(self):
        """重置环境"""
        node_state = self.node_env.reset()
        box_state = self.box_env.reset()
        self.steps = 0
        return {"node": node_state, "box": box_state}
    
    def step(self, action_dict):
        """执行动作
        
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
        else:
            node_next_state, node_reward, node_done = self.node_env.get_state(), 0, False
        
        # 执行边界框智能体的动作
        if "box" in action_dict:
            # 边界框环境使用节点环境的图结构
            self.box_env.G = self.node_env.G.copy() # 同步图结构
            box_next_state, box_reward, box_done = self.box_env.step(action_dict["box"])
        else:
            box_next_state, box_reward, box_done = self.box_env.get_state(), 0, False
        
        # 更新步数
        self.steps += 1
        
        print(f"node_reward: {node_reward}, box_reward: {box_reward}")
        # 合并奖励和完成状态
        reward = node_reward + box_reward
        done = node_done or box_done or self.steps >= self.max_steps
        
        return {"node": node_next_state, "box": box_next_state}, reward, done
    
    def get_state(self):
        """获取当前状态"""
        return {"node": self.node_env.get_state(), "box": self.box_env.get_state()}
