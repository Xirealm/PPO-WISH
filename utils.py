import torch
import numpy as np
import cv2
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
    return (bbox['min_x'] <= x <= bbox['max_x'] and bbox['min_y'] <= y <= bbox['max_y'])

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
        if (box['min_x'] <= x <= box['max_x'] and box['min_y'] <= y <= box['max_y']):
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
