import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Constants
IMAGE_SIZE = 560
DATASET = 'ISIC'
# DATASET = 'FSS-1000' 
CATAGORY = '100'
# CATAGORY = 'vineSnake'
# CATAGORY = 'bandedGecko'
BASE_DIR = os.path.dirname(__file__)

# Directory settings
RESULTS_DIR = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY,'masks')  
GROUND_TRUTH_DIR = os.path.join(BASE_DIR, 'dataset', DATASET, CATAGORY, 'target_masks')
time = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_DIR = os.path.join(BASE_DIR, 'evaluation', DATASET, CATAGORY, time)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def dice_coefficient(y_true, y_pred):
    """
    Calculate Dice Similarity Coefficient (DSC)
    
    Args:
    y_true: Ground truth binary mask
    y_pred: Predicted binary mask
    
    Returns:
    dice: Dice Similarity Coefficient
    """
    # Ensure inputs are binary masks
    y_true = y_true.astype(np.bool_)
    y_pred = y_pred.astype(np.bool_)
    
    # Calculate intersection
    intersection = np.logical_and(y_true, y_pred)
    
    # Calculate DSC
    dice = (2.0 * intersection.sum()) / (y_true.sum() + y_pred.sum() + 1e-7)
    
    return dice

def evaluate_segmentation(results_dir=RESULTS_DIR, ground_truth_dir=GROUND_TRUTH_DIR, output_dir=OUTPUT_DIR):
    result_files = sorted(glob(os.path.join(results_dir, "*.jpg")))
    
    if not result_files:
        raise FileNotFoundError(f"No mask images found in {results_dir}")
    
    results = []
    
    for result_file in tqdm(result_files, desc="Evaluating segmentation results"):
        # Get filename (remove _mask suffix and extension)
        filename = os.path.basename(result_file)
        # Find corresponding ground truth file
        gt_file = os.path.join(ground_truth_dir, filename)
        
        if not os.path.exists(gt_file):
            print(f"Warning: Ground truth not found for {filename}, skipping")
            continue
        
        # Read images
        result_img = cv2.imread(result_file, cv2.IMREAD_GRAYSCALE)
        gt_img = cv2.imread(gt_file, cv2.IMREAD_GRAYSCALE)
        
        # Ensure images are read correctly
        if result_img is None or gt_img is None:
            print(f"Warning: Cannot read images for {filename}, skipping")
            continue
        
        # Resize images to 560x560
        result_img = cv2.resize(result_img, (IMAGE_SIZE,IMAGE_SIZE))
        gt_img = cv2.resize(gt_img, (IMAGE_SIZE,IMAGE_SIZE))
        
        # Binarize images (if not already binary)
        if np.max(result_img) > 1:
            result_img = (result_img > 127).astype(np.uint8)
        if np.max(gt_img) > 1:
            gt_img = (gt_img > 127).astype(np.uint8)
        
        # Calculate DSC
        dice = dice_coefficient(gt_img, result_img)
        
        # Store results
        results.append({
            'filename': filename,
            'dice': dice
        })
    
    # Create DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate mean DSC
    mean_dice = results_df['dice'].mean()
    std_dice = results_df['dice'].std()
    
    print(f"Evaluation completed!")
    print(f"Mean DSC: {mean_dice:.4f} ± {std_dice:.4f}")
    print(f"Min DSC: {results_df['dice'].min():.4f}")
    print(f"Max DSC: {results_df['dice'].max():.4f}")
    
    # Plot DSC distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['dice'], bins=20, alpha=0.7, color='blue')
    plt.axvline(mean_dice, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_dice:.4f}')
    plt.title('DSC Distribution')
    plt.xlabel('DSC Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save results
    if output_dir:
        print(f"Saving results to {output_dir}")
        distribution_filename = "distribution.png"
        results_filename = "results.csv"
        summary_filename = "summary.txt"
        plt.savefig(os.path.join(output_dir, distribution_filename), dpi=300, bbox_inches='tight')
        results_df.to_csv(os.path.join(output_dir, results_filename), index=False)
        
        with open(os.path.join(output_dir, summary_filename), 'w') as f:
            f.write(f"Mean DSC: {mean_dice:.4f} +- {std_dice:.4f}\n")
            f.write(f"Min DSC: {results_df['dice'].min():.4f}\n")
            f.write(f"Max DSC: {results_df['dice'].max():.4f}\n")
    
    plt.show()

if __name__ == "__main__":
    evaluate_segmentation()