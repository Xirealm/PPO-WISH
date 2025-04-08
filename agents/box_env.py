import torch
import numpy as np
import networkx as nx
from utils import *

class BoxOptimizationEnv:
    def __init__(self, G, bbox_initial=None, image_size=560, max_steps=100, step_size=14):
        """Initialize the box optimization environment."""
        self.original_G = G.copy()
        self.G = G.copy()
        self.steps = 0
        self.max_steps = max_steps
        
        # Box configurations
        self.image_size = image_size
        self.step_size = step_size
        self.fine_step_size = self.step_size
        self.min_box_size = 56
        self.similarity_threshold = 0.8
        
        # Box tracking
        self.bbox_initial = bbox_initial if bbox_initial else []
        self.boxes = self.bbox_initial.copy()
        self.final_box = None
        
        # Previous state tracking
        self.previous_inner_coherence = 0
        self.previous_outer_difference = 0
        
        # Additional attributes
        self.features = None
        
        self.reset()

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
        patch_size = 14 
        
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
        coherence_change = normalize_distance(current_coherence, self.previous_inner_coherence)
        difference_change = normalize_distance(current_difference, self.previous_outer_difference)
            
        # 更新历史值
        self.previous_inner_coherence = current_coherence
        self.previous_outer_difference = current_difference
        
        print(f"Coherence: {current_coherence}, Difference: {current_difference}")
        
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
        if (box_idx >= len(self.boxes)):
            return self.get_state(), 0, True
            
        old_boxes = self.boxes.copy()
        
        reward = 0
        if operation == "merge" and params is not None:
            # 执行合并操作
            target_idx = params
            new_box_idx = self.merge_boxes(box_idx, target_idx)
            reward = self.calculate_box_reward(new_box_idx)
        else:
            if operation.startswith(("shrink_", "expand_")):
                # 根据操作类型确定移动方向
                direction = operation.split("_")[1]
                is_shrink = operation.startswith("shrink")
                amount = -self.step_size if is_shrink else self.step_size
                print(f"Box {box_idx} {operation} by {amount}")
                
                # 根据方向调整box大小
                if direction == "up":
                    new_val = self.boxes[box_idx]['min_y'] - amount  # 注意这里是减号，向上为负
                    if 0 <= new_val < self.boxes[box_idx]['max_y'] - self.min_box_size:
                        self.boxes[box_idx]['min_y'] = new_val
                elif direction == "down":
                    new_val = self.boxes[box_idx]['max_y'] + amount
                    if new_val > self.boxes[box_idx]['min_y'] + self.min_box_size and new_val <= self.image_size:
                        self.boxes[box_idx]['max_y'] = new_val
                elif direction == "left":
                    new_val = self.boxes[box_idx]['min_x'] - amount  # 注意这里是减号，向左为负
                    if 0 <= new_val < self.boxes[box_idx]['max_x'] - self.min_box_size:
                        self.boxes[box_idx]['min_x'] = new_val
                elif direction == "right":
                    new_val = self.boxes[box_idx]['max_x'] + amount
                    if new_val > self.boxes[box_idx]['min_x'] + self.min_box_size:
                        self.boxes[box_idx]['max_x'] = new_val
                
                reward = self.calculate_box_reward(box_idx)
        
        print(f"Operation: {operation}, Reward: {reward}")
        # 如果奖励为负，回滚操作
        # if reward < 0:
        #     self.boxes = old_boxes
        #     reward = 0
        # else:
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