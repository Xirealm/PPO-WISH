import os
import sys
import time
import warnings
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Set paths for feature matching and segmentation modules
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

# from segment_anything import sam_model_registry, SamPredictor
from segmenter.segment import process_image, loading_seg, seg_main, show_points
from feature_matching.generate_points import generate, loading_dino, distance_calculate
from test_GPOA import test_agent, optimize_nodes
from utils import  NodeOptimizationEnv, NodeAgent, BoxAgent, MultiAgentEnv, calculate_distances, convert_to_edges, calculate_center_points, refine_mask, BoxOptimizationEnv

# Ignore all warnings
warnings.filterwarnings("ignore")

# DATASET = 'FSS-1000' 
DATASET = 'ISIC'
# DATASET = 'Kvasir'
CATAGORY = '100'
# CATAGORY = '100'
# CATAGORY = 'vineSnake'
# CATAGORY = 'bandedGecko'

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 560

# Define paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, 'dataset', DATASET, CATAGORY)
REFERENCE_IMAGE_DIR = os.path.join(DATA_DIR, 'reference_images')
MASK_DIR = os.path.join(DATA_DIR, 'reference_masks')
Q_TABLE_PATH = os.path.join(BASE_DIR, 'model', 'best_q_table.pkl')
IMAGE_DIR = os.path.join(DATA_DIR, 'target_images')  # Path for test images
RESULTS_DIR = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY)
SAVE_DIR = os.path.join(RESULTS_DIR, 'masks')
FINAL_PROMPTS_DIR = os.path.join(RESULTS_DIR, 'final_prompts_image')  # 新增保存最终提示的目录

# Ensure the results directories exist
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(FINAL_PROMPTS_DIR, exist_ok=True)  # 创建新目录

# Load models for segmentation and feature generation
def load_models():
    """
    Load the segmentation model and DINO feature extractor.
    """
    try:
        model_seg = loading_seg('vitl', DEVICE)
        model_dino = loading_dino(DEVICE)
        return model_seg, model_dino
    except Exception as e:
        print(f"Error loading models: {e}")
        sys.exit(1)

# Load multi-agent system
def load_agents():
    """
    Load node and box agents.
    """
    try:
        node_env = NodeOptimizationEnv
        box_env = BoxOptimizationEnv
        node_agent = NodeAgent(node_env)
        box_agent = BoxAgent(box_env)
        
        # Load trained models
        node_agent.q_table = torch.load(os.path.join(BASE_DIR, 'model', 'node_best_q_table.pkl'), weights_only=False)
        box_agent.q_table = torch.load(os.path.join(BASE_DIR, 'model', 'box_best_q_table.pkl'), weights_only=False)
        
        return node_agent, box_agent
    except Exception as e:
        print(f"Error loading agents: {e}")
        # Fallback to using only node agent
        node_env = NodeOptimizationEnv
        node_agent = NodeAgent(node_env)
        node_agent.q_table = torch.load(Q_TABLE_PATH, weights_only=False)
        return node_agent, None

# Process a single image
def process_single_image(node_agent, box_agent, model_dino, model_seg, image_name, reference, mask_dir):
    """
    Use multi-agent system to process a single image for segmentation and optimization.

    Parameters:
    - node_agent: Node agent
    - box_agent: Box agent
    - model_dino: DINO feature extraction model
    - model_seg: SAM model
    - image_name: Name of the image to process
    - reference: Reference image for feature comparison
    - mask_dir: Directory containing ground truth masks
    """
    try:
        # Load input image and reference data
        image_path = os.path.join(IMAGE_DIR, image_name)
        image = Image.open(image_path).resize((IMAGE_SIZE, IMAGE_SIZE))
        reference_image = Image.open(os.path.join(REFERENCE_IMAGE_DIR, reference)).resize((IMAGE_SIZE, IMAGE_SIZE))
        gt_mask = Image.open(os.path.join(mask_dir, reference)).resize((IMAGE_SIZE, IMAGE_SIZE))

        # Generate features and initial positive/negative prompts
        image_inner = [reference_image, image]
        start_time = time.time()
        features, pos_indices, neg_indices = generate(gt_mask, image_inner, DEVICE, model_dino, IMAGE_SIZE)
        # unique indices
        pos_indices = torch.unique(pos_indices).to(DEVICE)
        neg_indices = torch.unique(neg_indices).to(DEVICE)
        # Remove intersections
        intersection = set(pos_indices.tolist()).intersection(set(neg_indices.tolist()))
        if intersection:
            pos_indices = torch.tensor([x for x in pos_indices.cpu().tolist() if x not in intersection]).cuda()
            neg_indices = torch.tensor([x for x in neg_indices.cpu().tolist() if x not in intersection]).cuda()
        end_time = time.time()
        print(f"Time to generate initial prompts: {end_time - start_time:.4f} seconds")

        if len(pos_indices) != 0 and len(neg_indices) != 0:
            # 使用聚类生成的多个box
            from feature_matching.generate_box import generate_boxes
            boxes, merged_box = generate_boxes(features, pos_indices)
            
            # 转换box格式
            box_data = [
                {
                    'min_x': int(box[0]), 'min_y': int(box[1]),
                    'max_x': int(box[2]), 'max_y': int(box[3])
                }
                for box in boxes
            ]

            # Calculate feature distances and physical distances
            feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
                features, pos_indices, neg_indices, IMAGE_SIZE, DEVICE)

            # Convert to edge representation
            feature_pos_edge = convert_to_edges(pos_indices, pos_indices, feature_pos_distances)
            physical_pos_edge = convert_to_edges(pos_indices, pos_indices, physical_pos_distances)
            physical_neg_edge = convert_to_edges(neg_indices, neg_indices, physical_neg_distances)
            feature_cross_edge = convert_to_edges(pos_indices, neg_indices, feature_cross_distances)
            physical_cross_edge = convert_to_edges(pos_indices, neg_indices, physical_cross_distances)

            # Create graph structure
            G = nx.MultiGraph()
            G.add_nodes_from(pos_indices.cpu().numpy(), category='pos')
            G.add_nodes_from(neg_indices.cpu().numpy(), category='neg')

            # Add weighted edges
            G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
            G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
            G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
            G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
            G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

            # 使用多box系统优化
            if box_agent is not None:
                multi_env = MultiAgentEnv(G, box_data, IMAGE_SIZE, max_steps=100)
                node_agent.env = multi_env.node_env
                box_agent.env = multi_env.box_env
                
                # 设置特征信息
                multi_env.box_env.features = features
                
                state = multi_env.reset()
                done = False
                
                # Alternate execution of node agent and box agent actions
                step_count = 0
                while not done:
                    if step_count % 2 == 0:  # Even step executes node agent action
                        node_action = node_agent.get_action(state["node"])
                        action_dict = {"node": node_action}
                    else:  # Odd step executes box agent action
                        box_action = box_agent.get_action(state["box"])
                        action_dict = {"box": box_action}
                    
                    next_state, _, done = multi_env.step(action_dict)
                    state = next_state
                    step_count += 1
                
                # Get optimized prompts and bounding box
                opt_pos_indices = torch.tensor([node for node in multi_env.node_env.pos_nodes])
                opt_neg_indices = torch.tensor([node for node in multi_env.node_env.neg_nodes])
                optimized_boxes = multi_env.box_env.boxes
                
                # 合并所有优化后的box为最终box
                final_box = {
                    'min_x': min(box['min_x'] for box in optimized_boxes),
                    'min_y': min(box['min_y'] for box in optimized_boxes),
                    'max_x': max(box['max_x'] for box in optimized_boxes),
                    'max_y': max(box['max_y'] for box in optimized_boxes)
                }
                
                # Generate points first 
                pos_points = calculate_center_points(opt_pos_indices, IMAGE_SIZE)
                neg_points = calculate_center_points(opt_neg_indices, IMAGE_SIZE)

                # 可视化优化后的点和boxes
                plt.figure(figsize=(10, 10))
                plt.imshow(image)
                
                # 显示点和所有小box
                coords = np.array(pos_points + neg_points)
                labels = np.concatenate([
                    np.ones(len(pos_points)),
                    np.zeros(len(neg_points))
                ])
                show_points(coords, labels, plt.gca())
                
                # 绘制所有优化后的小box
                for i, box in enumerate(optimized_boxes):
                    rect = plt.Rectangle(
                        (box['min_x'], box['min_y']),
                        box['max_x'] - box['min_x'],
                        box['max_y'] - box['min_y'],
                        fill=False,
                        edgecolor=plt.cm.Set3(i / len(optimized_boxes)),
                        linewidth=2.0
                    )
                    plt.gca().add_patch(rect)
                
                # 绘制最终合并的box
                rect = plt.Rectangle(
                    (final_box['min_x'], final_box['min_y']),
                    final_box['max_x'] - final_box['min_x'],
                    final_box['max_y'] - final_box['min_y'],
                    fill=False,
                    edgecolor='#f08c00',
                    linewidth=2.5
                )
                plt.gca().add_patch(rect)
                
                # 将最终box添加到优化结果中
                multi_env.box_env.final_box = final_box
            else:
                # Use only node agent
                opt_pos_indices, opt_neg_indices = optimize_nodes(
                    node_agent, pos_indices, neg_indices, features, max_steps=100, device=DEVICE, image_size=IMAGE_SIZE
                )
            
            end_time = time.time()
            print(f"len(opt_pos_indices): {len(opt_pos_indices)}, len(opt_neg_indices): {len(opt_neg_indices)}")
            print(f"Time to optimize prompts: {end_time - start_time:.4f} seconds")

            # Perform segmentation
            mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg)

            # Refine the mask before saving
            refined_mask = refine_mask(mask, threshold=0.5)
            mask = Image.fromarray(refined_mask)
            mask.save(os.path.join(SAVE_DIR, f"{image_name}"))

            # Save final prompt points and bounding box visualization image
            plt.axis('off')
            plt.savefig(os.path.join(FINAL_PROMPTS_DIR, f'{image_name}_final.png'), 
                       bbox_inches='tight', 
                       pad_inches=0)
            plt.close()

        else:
            print(f"Skipping {image_name}: No positive or negative indices found.")
    except Exception as e:
        print(f"Error processing {image_name}: {e}")

# Main function
if __name__ == "__main__":
    # Load models
    model_seg, model_dino = load_models()

    # Load multi-agent system
    node_agent, box_agent = load_agents()

    # Get reference image list
    reference_list = os.listdir(REFERENCE_IMAGE_DIR)
    if not reference_list:
        print("No reference images found.")
        sys.exit(1)

    # Use the first reference image
    reference = reference_list[0]

    # Process all images in the test directory
    img_list = os.listdir(IMAGE_DIR)
    for img_name in tqdm(img_list, desc="Processing images"):
        process_single_image(node_agent, box_agent, model_dino, model_seg, img_name, reference, MASK_DIR)

    print("Processing complete!")
