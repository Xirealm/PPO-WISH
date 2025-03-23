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
from utils import generate_points, NodeOptimizationEnv, NodeAgent, BoxAgent, MultiAgentEnv, calculate_distances, convert_to_edges, calculate_bounding_box, calculate_center_points, refine_mask, get_box_node_indices, BoxOptimizationEnv

# Ignore all warnings
warnings.filterwarnings("ignore")

# DATASET = 'FSS-1000' 
DATASET = 'ISIC'
# DATASET = 'Kvasir'
CATAGORY = '10'
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
            # Generate bounding box for positive points
            pos_points = calculate_center_points(pos_indices, IMAGE_SIZE)
            initial_bbox = calculate_bounding_box(pos_points, patch_size=14, image_size=IMAGE_SIZE)
            
            if initial_bbox is not None:
                min_x, min_y, max_x, max_y = initial_bbox
                bbox_data = {'min_x': min_x, 'min_y': min_y, 'max_x': max_x, 'max_y': max_y}
                print(f"Generated bounding box: ({min_x}, {min_y}, {max_x}, {max_y})")
                input_box = np.array([min_x, min_y, max_x, max_y])
            else:
                print("No valid bounding box generated")
                center = IMAGE_SIZE // 2
                size = IMAGE_SIZE // 4
                bbox_data = {
                    'min_x': center - size // 2,
                    'min_y': center - size // 2,
                    'max_x': center + size // 2,
                    'max_y': center + size // 2
                }
                input_box = np.array([bbox_data['min_x'], bbox_data['min_y'], 
                                      bbox_data['max_x'], bbox_data['max_y']])

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

            # Use multi-agent system to optimize prompts
            start_time = time.time()
            if box_agent is not None:
                # Use multi-agent system
                multi_env = MultiAgentEnv(G, bbox_data, IMAGE_SIZE, max_steps=100)
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
                opt_bbox = multi_env.box_env.bbox
                input_box = np.array([opt_bbox['min_x'], opt_bbox['min_y'], 
                                      opt_bbox['max_x'], opt_bbox['max_y']])
            else:
                # Use only node agent
                opt_pos_indices, opt_neg_indices = optimize_nodes(
                    node_agent, pos_indices, neg_indices, features, max_steps=100, device=DEVICE, image_size=IMAGE_SIZE
                )
            
            end_time = time.time()
            print(f"len(opt_pos_indices): {len(opt_pos_indices)}, len(opt_neg_indices): {len(opt_neg_indices)}")
            print(f"Time to optimize prompts: {end_time - start_time:.4f} seconds")

            # Generate points and perform segmentation
            pos_points, neg_points = generate_points(opt_pos_indices, opt_neg_indices, IMAGE_SIZE)
            
            # Add box prompt to segmentation
            mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg, input_box=input_box)

            # Refine the mask before saving
            refined_mask = refine_mask(mask, threshold=0.5)
            mask = Image.fromarray(refined_mask)
            mask.save(os.path.join(SAVE_DIR, f"{image_name}"))

            # Save final prompt points and bounding box visualization image
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            # Prepare points and labels for visualization
            coords = np.array(pos_points + neg_points)
            labels = np.concatenate([
                np.ones(len(pos_points)),
                np.zeros(len(neg_points))
            ])
            
            # Display points
            show_points(coords, labels, plt.gca())
            
            # Draw bounding box
            rect = plt.Rectangle((input_box[0], input_box[1]),
                               input_box[2] - input_box[0],
                               input_box[3] - input_box[1],
                               fill=False,
                               edgecolor='#f08c00',
                               linewidth=2.5)
            plt.gca().add_patch(rect)
            
            # Remove axes and save image
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
