import torch
import numpy as np
from sklearn.cluster import DBSCAN

def get_patch_coordinates(indices, image_size=560, patch_size=14):
    """
    计算patch在物理空间中的中心点坐标
    """
    rows = indices.cpu().numpy() // (image_size // patch_size)
    cols = indices.cpu().numpy() % (image_size // patch_size)
    centers_x = (cols * patch_size + patch_size // 2).astype(int)
    centers_y = (rows * patch_size + patch_size // 2).astype(int)
    return np.column_stack((centers_x, centers_y))

def cluster_features(features, pos_indices, eps=28, min_samples=3):
    """
    对所有patch进行聚类，并基于正样本识别目标相关的cluster
    """
    # 获取所有patch的特征和坐标
    all_features = features[1]
    num_patches = all_features.shape[0]
    all_indices = torch.arange(num_patches)
    all_coordinates = get_patch_coordinates(all_indices)
    
    # 使用DBSCAN在物理空间进行聚类
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(all_coordinates)
    labels = clustering.labels_
    
    # 组织聚类结果
    unique_labels = np.unique(labels)
    print(len(unique_labels))
    clustered_indices = []
    pos_indices_set = set(pos_indices.cpu().numpy())
    
    for label in unique_labels:
        if label != -1:  # 排除噪声点
            cluster_mask = (labels == label)
            cluster_indices = all_indices[cluster_mask]
            
            # 计算cluster中包含的正样本比例
            pos_in_cluster = set(cluster_indices.numpy()) & pos_indices_set
            pos_ratio = len(pos_in_cluster) / len(cluster_indices)
            
            print(pos_ratio)
            # 只有当cluster包含足够比例的正样本时才保留
            if pos_ratio >= 0.1:  # 可调整的阈值
                # 验证特征空间的相似性
                cluster_features = all_features[cluster_indices]
                pos_mean_feature = torch.mean(all_features[list(pos_indices_set)], dim=0)
                distances = torch.norm(cluster_features - pos_mean_feature, dim=1)
                
                # 保留特征相似的点
                valid_indices = distances < 0.5  # 可调整的特征相似性阈值
                print(valid_indices)
                if torch.sum(valid_indices) >= min_samples:
                    clustered_indices.append(cluster_indices[valid_indices])
    
    return clustered_indices

def calculate_box_from_indices(indices, image_size=560, patch_size=14, padding=14):
    """
    计算更紧凑的边界框
    """
    coordinates = get_patch_coordinates(indices)
    
    min_x = max(0, np.min(coordinates[:, 0]) - padding)
    min_y = max(0, np.min(coordinates[:, 1]) - padding)
    max_x = min(image_size, np.max(coordinates[:, 0]) + padding)
    max_y = min(image_size, np.max(coordinates[:, 1]) + padding)
    
    # 确保边界框的最小尺寸
    if max_x - min_x < patch_size:
        center_x = (max_x + min_x) // 2
        min_x = max(0, center_x - patch_size)
        max_x = min(image_size, center_x + patch_size)
    
    if max_y - min_y < patch_size:
        center_y = (max_y + min_y) // 2
        min_y = max(0, center_y - patch_size)
        max_y = min(image_size, center_y + patch_size)
    
    return (min_x, min_y, max_x, max_y)

def merge_boxes(boxes):
    """
    合并多个边界框为一个
    """
    if not boxes:
        return None
        
    min_x = min(box[0] for box in boxes)
    min_y = min(box[1] for box in boxes)
    max_x = max(box[2] for box in boxes)
    max_y = max(box[3] for box in boxes)
    
    return (min_x, min_y, max_x, max_y)

def generate_boxes(features, pos_indices):
    """
    生成基于局部特征的边界框
    """
    # 特征聚类
    clustered_indices = cluster_features(features, pos_indices)
    
    # 为每个聚类生成边界框
    cluster_boxes = []
    for indices in clustered_indices:
        box = calculate_box_from_indices(indices)
        cluster_boxes.append(box)
    
    # 合并所有边界框
    merged_box = merge_boxes(cluster_boxes) if cluster_boxes else None
    
    return cluster_boxes, merged_box
