import numpy as np
import networkx as nx
import random
from config import *
from utils import *

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
        
        self.features = None  # 初始化特征

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
        
        # if self.min_nodes < len(self.pos_nodes) < self.max_nodes and self.min_nodes < len(self.neg_nodes) < self.max_nodes:
        #     # 如果reward为负，回滚操作
        #     if reward < 0:
        #         self.revertStep(action)
        #         reward = self.calculate_reward(operation)
        
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
            reward += 5 * (self.previous_feature_pos_mean - mean_feature_pos)
        else:
            reward -= 5 * (mean_feature_pos - self.previous_feature_pos_mean)

        if mean_feature_cross > self.previous_feature_cross_mean:
            reward += 5 * (mean_feature_cross - self.previous_feature_cross_mean)
        else:
            reward -= 5 * (self.previous_feature_cross_mean - mean_feature_cross)

        if mean_physical_pos > self.previous_physical_pos_mean:
            reward += 5 * (mean_physical_pos - self.previous_physical_pos_mean)
        else:
            reward -= 5 *(self.previous_physical_pos_mean - mean_physical_pos)

        if mean_physical_neg > self.previous_physical_neg_mean:
            reward += 1 * (mean_physical_neg - self.previous_physical_neg_mean)
        else:
            reward -= 1 *(self.previous_physical_neg_mean - mean_physical_neg)

        if mean_physical_cross < self.previous_physical_cross_mean:
            reward += 1 * (self.previous_physical_cross_mean - mean_physical_cross)
        else:
            reward -= 1 * (mean_physical_cross - self.previous_physical_cross_mean)
            
        # if operation == "add":
        #     # 根据add操作后的状态变化调整惩罚
        #     if (mean_feature_pos < self.previous_feature_pos_mean and 
        #         mean_feature_cross > self.previous_feature_cross_mean):
        #         # add操作改善了特征距离，给予较小惩罚
        #         reward -= 3
        #     else:
        #         # add操作没有改善特征距离，给予较大惩罚
        #         reward -= 5

        # 初始化位置相关奖励
        position_reward = 0
        
        # 获取在box内外的节点信息
        box_info = get_box_node_indices(self.G, self.boxes)
        outside_pos_indices = box_info[3]  # 获取box外正提示点
        inside_neg_indices = box_info[4]  # 获取box内负提示点
        
        # 根据操作类型和位置给予不同的奖励
        if operation == "restore_pos":
            if len(outside_pos_indices) > 0:
                position_reward -= 2 * (outside_pos_indices / len(self.G.nodes())) 
                
        if operation == "restore_neg":
            if len(inside_neg_indices) > 0:
                position_reward -= 2 * (outside_pos_indices / len(self.G.nodes())) 
                
        # 对每个box内的负样本数量进行检查和惩罚
        # box_counts = count_nodes_per_box(self.G, self.boxes)
        # for pos_count, neg_count in box_counts:
        #     if pos_count == 1 or neg_count == 1:
        #         position_reward += 1 * pos_count
        #     if pos_count >= 2 or neg_count >= 2:
        #         position_reward -= 1 * pos_count

        print(f"Operation: {operation}, Position Reward: {position_reward}")
        reward += position_reward

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