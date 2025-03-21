import os
import sys
import time
import warnings
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Set paths for feature matching and segmentation modules
generate_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'feature_matching'))
segment_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'segmenter'))
sys.path.append(segment_path)
sys.path.append(generate_path)

# from segment_anything import sam_model_registry, SamPredictor
from segmenter.segment import process_image, loading_seg, seg_main, show_points
from feature_matching.generate_points import generate, loading_dino, distance_calculate
from test_GPOA import test_agent, optimize_nodes
from utils import generate_points, GraphOptimizationEnv, QLearningAgent, calculate_distances, convert_to_edges, calculate_bounding_box, calculate_center_points,refine_mask

# Ignore all warnings
warnings.filterwarnings("ignore")

SIZE = 560
# DATASET = 'FSS-1000' 
DATASET = 'ISIC'
# DATASET = 'Kvasir'
# CATAGORY = '10'
CATAGORY = '10'
# CATAGORY = 'vineSnake'
# CATAGORY = 'bandedGecko'

# Set device (GPU if available, otherwise CPU)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = SIZE

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

# Process a single image
def process_single_image(agent, model_dino, model_seg, image_name, reference, mask_dir):
    """
    Process a single image for segmentation and optimization.

    Parameters:
    - agent: Q-learning agent for optimization
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
        end_time = time.time()
        print(f"Time to generate initial prompts: {end_time - start_time:.4f} seconds")

        if len(pos_indices) != 0 and len(neg_indices) != 0:
            # Generate bounding box for positive points
            pos_points = calculate_center_points(pos_indices, IMAGE_SIZE)
            bbox = calculate_bounding_box(pos_points, patch_size=14, image_size=IMAGE_SIZE)
            
            if bbox is not None:
                min_x, min_y, max_x, max_y = bbox
                print(f"Generated bounding box: ({min_x}, {min_y}, {max_x}, {max_y})")
                input_box = np.array([min_x, min_y, max_x, max_y])
            else:
                print("No valid bounding box generated")
                input_box = None

            # Optimize prompts using Q-learning
            start_time = time.time()
            opt_pos_indices, opt_neg_indices = optimize_nodes(
                agent, pos_indices, neg_indices, features, max_steps=100, device=DEVICE, image_size=IMAGE_SIZE
            )
            end_time = time.time()
            print(f"len(opt_pos_indices): {len(opt_pos_indices)}, len(opt_neg_indices): {len(opt_neg_indices)}")
            print(f"Time to optimize prompts: {end_time - start_time:.4f} seconds")

            # Generate points and perform segmentation
            pos_points, neg_points = generate_points(opt_pos_indices, opt_neg_indices, IMAGE_SIZE)
            
            # Add box prompt to segmentation
            if input_box is not None:
                mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg, input_box=input_box)
            else:
                mask = seg_main(image, pos_points, neg_points, DEVICE, model_seg)

            # Refine the mask before saving
            refined_mask = refine_mask(mask,threshold=0.5)
            mask = Image.fromarray(refined_mask)
            mask.save(os.path.join(SAVE_DIR, f"{image_name}"))

            # 保存最终提示点和边界框的可视化图像
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            
            # 准备点和标签用于可视化
            coords = np.array(pos_points + neg_points)
            labels = np.concatenate([
                np.ones(len(pos_points)),
                np.zeros(len(neg_points))
            ])
            
            # 显示点
            show_points(coords, labels, plt.gca())
            
            # 如果有边界框，则绘制边界框
            if input_box is not None:
                rect = plt.Rectangle((input_box[0], input_box[1]),
                                   input_box[2] - input_box[0],
                                   input_box[3] - input_box[1],
                                   fill=False,
                                   edgecolor='#f08c00',
                                   linewidth=2.5)
                plt.gca().add_patch(rect)
            
            # 移除坐标轴并保存图像
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

    # Initialize Q-learning agent
    env = GraphOptimizationEnv
    agent = QLearningAgent(env)
    agent.q_table = torch.load(Q_TABLE_PATH,weights_only=False)

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
        process_single_image(agent, model_dino, model_seg, img_name, reference, MASK_DIR)

    print("Processing complete!")
