import torch
import os

# 设备配置
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 图像配置 
IMAGE_SIZE = 560  # Must be a multiple of 14
PATCH_SIZE = 14

# MIN_NODES = 5
# MAX_NODES = 20

# # 数据集相关配置
# DATASET_NAME = 'FSS-1000'

# # 路径配置
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_DIR = os.path.join(BASE_DIR, 'dataset', DATASET_NAME)
# REFERENCE_IMAGE_DIR = os.path.join(DATASET_DIR, 'reference_images')
# REFERENCE_MASK_DIR = os.path.join(DATASET_DIR, 'reference_masks')
# TARGET_IMAGE_DIR = os.path.join(DATASET_DIR, 'target_images')
# RESULTS_DIR = os.path.join(BASE_DIR, 'results', DATASET_NAME)

# # 模型配置
# MODEL_CHECKPOINT_DIR = os.path.join(BASE_DIR, 'checkpoints')

# # Q-learning 相关配置
# QLEARNING_CONFIG = {
#     'alpha': 0.1,  # 学习率
#     'gamma': 0.9,  # 折扣因子
#     'epsilon_start': 1.0,  # 初始探索率
#     'epsilon_end': 0.1,  # 最终探索率
#     'epsilon_decay': 0.995,  # 探索率衰减
#     'memory_size': 10000,  # 记忆容量
#     'batch_size': 64,  # 批次大小
#     'reward_threshold': 0.1,  # 奖励阈值
# }
