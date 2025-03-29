import torch
import numpy as np
from sklearn.cluster import KMeans

def cluster_features(features, pos_indices, n_clusters=5):
    """
    对所有patch特征进行聚类，然后找出与正样本相关的cluster
    """
    # 获取所有patch的特征
    all_features = features[1].cpu().numpy()
    total_patches = all_features.shape[0]
    
    # 使用K-means聚类所有特征
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    all_labels = kmeans.fit_predict(all_features)
    
    # 获取正样本所在的cluster标签
    pos_labels = all_labels[pos_indices.cpu()]
    unique_pos_labels = np.unique(pos_labels)
    
    # 选择包含正样本比例较高的cluster
    selected_clusters = []
    for label in unique_pos_labels:
        cluster_indices = np.where(all_labels == label)[0]
        pos_in_cluster = np.intersect1d(cluster_indices, pos_indices.cpu().numpy())
        
        # 如果cluster中正样本比例超过阈值，则选择该cluster
        if len(pos_in_cluster) / len(cluster_indices) > 0.1:  # 阈值可调整
            selected_clusters.append(torch.tensor(cluster_indices).cuda())
            
    return selected_clusters

def calculate_box_from_indices(indices, image_size=560, patch_size=14, padding=21):
    """
    根据patch索引计算边界框
    """
    rows = indices.cpu().numpy() // (image_size // patch_size)
    cols = indices.cpu().numpy() % (image_size // patch_size)
    
    # 计算patch中心点坐标
    points_x = (cols * patch_size + patch_size // 2).astype(int)
    points_y = (rows * patch_size + patch_size // 2).astype(int)
    
    # 计算边界框
    min_x = max(0, np.min(points_x) - padding)
    min_y = max(0, np.min(points_y) - padding)
    max_x = min(image_size, np.max(points_x) + padding)
    max_y = min(image_size, np.max(points_y) + padding)
    
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

def generate_boxes(features, pos_indices, n_clusters=5):
    """
    生成聚类边界框和合并后的边界框
    """
    # 特征聚类
    clustered_indices = cluster_features(features, pos_indices, n_clusters)
    
    # 为每个聚类生成边界框
    cluster_boxes = []
    for indices in clustered_indices:
        # 只处理同时包含正样本的indices
        if len(np.intersect1d(indices.cpu().numpy(), pos_indices.cpu().numpy())) > 0:
            box = calculate_box_from_indices(indices)
            cluster_boxes.append(box)
    
    # 合并所有边界框
    merged_box = merge_boxes(cluster_boxes)
    
    return cluster_boxes, merged_box
