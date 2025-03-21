import os
import torch
import numpy as np
from PIL import Image
import cv2
import networkx as nx
import pandas as pd
import random
from collections import defaultdict, deque
import time
from datetime import timedelta, datetime
from utils import GraphOptimizationEnv, QLearningAgent

# Constants
SIZE = 560
DATASET = 'ISIC'
CATAGORY = '10'
BASE_DIR = os.path.dirname(__file__)

def main():
    """Main function to train the Q-learning agent."""
    global device, image_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = SIZE

    # 设置初始提示数据目录路径
    base_dir = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY, 'initial_prompts')
    # 获取目录下所有文件
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    # 提取文件名前缀（前三个下划线分隔的部分）
    file_prefixes = set('_'.join(f.split('_')[:3]) for f in files)
    # 训练参数设置
    max_steps = 100 # 每个episode的最大步数
    env = GraphOptimizationEnv # 初始化环境
    agent = QLearningAgent(env) # 初始化Q-learning agent
    # 创建输出目录
    current_time = datetime.now().strftime("%Y%m%d_%H%M") # 当前时间
    output_path = os.path.join(os.path.dirname(__file__), 'train', current_time) # 输出目录
    # 训练智能体
    rewards = agent.train(episodes=30, output_path=output_path, base_dir=base_dir, file_prefixes=file_prefixes, max_steps=max_steps)
     # 绘制训练奖励曲线
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Rewards Over Episodes')
    plt.savefig(f"{output_path}/rewards_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
