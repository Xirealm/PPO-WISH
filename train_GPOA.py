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
from utils import NodeOptimizationEnv, NodeAgent, BoxAgent, MultiAgentEnv, BoxOptimizationEnv, calculate_distances, convert_to_edges, get_box_node_indices
import json

# Constants
SIZE = 560
DATASET = 'ISIC'
CATAGORY = '10'
BASE_DIR = os.path.dirname(__file__)

def train_multi_agent(node_agent, box_agent, episodes, output_path, base_dir, file_prefixes, max_steps):
    """训练多智能体系统
    Args:
        node_agent: 节点智能体
        box_agent: 矩形框智能体
        episodes: 训练回合数
        output_path: 输出路径
        base_dir: 训练数据目录 
        file_prefixes: 文件前缀列表
        max_steps: 每回合最大步数
    Returns:
        rewards: 每回合的奖励列表
    """
    rewards = []
    image_size = SIZE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_path, exist_ok=True)  # 创建输出目录
    start_time = time.time()  # 记录开始时间

    # 训练循环
    for episode in range(episodes):
        # 随机选择一个文件前缀
        selected_prefix = random.choice(list(file_prefixes))
        print(f"Episode {episode + 1}/{episodes}, Selected file prefix: {selected_prefix}")
        
        # 构建特征文件和正负索引文件的完整路径
        feature_file = os.path.join(base_dir, f"{selected_prefix}_features.pt")
        pos_file = os.path.join(base_dir, f"{selected_prefix}_initial_indices_pos.pt")
        neg_file = os.path.join(base_dir, f"{selected_prefix}_initial_indices_neg.pt")
        bbox_file = os.path.join(base_dir, f"{selected_prefix}_bbox.json") 
        
        # 检查所需文件是否存在
        if not (os.path.exists(feature_file) and os.path.exists(pos_file) and os.path.exists(neg_file) and os.path.exists(bbox_file)):
            print(f"Required files not found for prefix {selected_prefix}, skipping this episode.")
            continue
        
        # 加载特征和正负索引数据
        features = torch.load(feature_file, weights_only=True).to(device)
        positive_indices = torch.load(pos_file, weights_only=True).to(device)
        negative_indices = torch.load(neg_file, weights_only=True).to(device)
        # 加载bbox数据
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)
            
        # 确保索引唯一
        positive_indices = torch.unique(positive_indices).to(device)
        negative_indices = torch.unique(negative_indices).to(device)
        
        # 移除正负索引的交集
        set1 = set(positive_indices.tolist())
        set2 = set(negative_indices.tolist())
        intersection = set1.intersection(set2)
        if intersection:
            positive_indices = torch.tensor([x for x in positive_indices.cpu().tolist() if x not in intersection]).cuda()
            negative_indices = torch.tensor([x for x in negative_indices.cpu().tolist() if x not in intersection]).cuda()
            
        # 检查正/负索引是否为空，空则跳过
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        print(f"Positive indices: {positive_indices.shape}, Negative indices: {negative_indices.shape}")

        # 计算特征距离和物理距离
        feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
            features, positive_indices, negative_indices, image_size, device)

        # 转换为边的表示形式
        feature_pos_edge = convert_to_edges(positive_indices, positive_indices, feature_pos_distances)
        physical_pos_edge = convert_to_edges(positive_indices, positive_indices, physical_pos_distances)
        physical_neg_edge = convert_to_edges(negative_indices, negative_indices, physical_neg_distances)
        feature_cross_edge = convert_to_edges(positive_indices, negative_indices, feature_cross_distances)
        physical_cross_edge = convert_to_edges(positive_indices, negative_indices, physical_cross_distances)

        # 创建图结构
        G = nx.MultiGraph()
        G.add_nodes_from(positive_indices.cpu().numpy(), category='pos')
        G.add_nodes_from(negative_indices.cpu().numpy(), category='neg')

        # 添加带权重的边
        G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
        G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
        G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
        G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
        G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

        # 获取列表格式的结果
        (inside_indices, outside_indices,
        inside_pos_indices, outside_pos_indices,
        inside_neg_indices, outside_neg_indices) = get_box_node_indices(G, bbox_data)
        
        print(f"Inside bbox - Pos: {len(inside_pos_indices)}, Neg: {len(inside_neg_indices)}")
        print(f"Outside bbox - Pos: {len(outside_pos_indices)}, Neg: {len(outside_neg_indices)}")

        # 初始化多智能体环境
        multi_env = MultiAgentEnv(G, bbox_data, image_size, max_steps)
        node_agent.env = multi_env.node_env
        box_agent.env = multi_env.box_env
        
        # 设置特征信息
        multi_env.box_env.features = features 
        
        state = multi_env.reset()
        done = False
        total_reward = 0

        # 根据最佳奖励动态调整epsilon
        normalized_reward = (node_agent.best_reward - 0) / (5 - 0)
        if node_agent.best_reward < 0:
            node_agent.epsilon = node_agent.epsilon_start
            box_agent.epsilon = box_agent.epsilon_start
        elif node_agent.best_reward >= 5:
            node_agent.epsilon = node_agent.epsilon_end
            box_agent.epsilon = box_agent.epsilon_end
        else:
            node_agent.epsilon = 1 - normalized_reward
            box_agent.epsilon = 1 - normalized_reward
        print(f"best_reward:{node_agent.best_reward},node_epsilon:{node_agent.epsilon},box_epsilon:{box_agent.epsilon}")

        # 训练循环 - 交替执行节点智能体和矩形框智能体的动作
        step_count = 0
        while not done:
            # 在每个时间步，决定要执行的智能体
            if step_count % 2 == 0:  # 偶数步执行节点智能体动作
                node_action = node_agent.get_action(state["node"])
                action_dict = {"node": node_action}
                next_state, reward, done = multi_env.step(action_dict)
                
                # 更新节点智能体Q表和记忆
                node_agent.memory.append((state["node"], node_action, reward, next_state["node"]))
                node_agent.update_q_table(state["node"], node_action, reward, next_state["node"])
                node_agent.replay()
            else:  # 奇数步执行矩形框智能体动作
                box_action = box_agent.get_action(state["box"])
                action_dict = {"box": box_action}
                print(f"Box action: {box_action}")
                next_state, reward, done = multi_env.step(action_dict)
                
                # 更新矩形框智能体Q表和记忆
                box_agent.memory.append((state["box"], box_action, reward, next_state["box"]))
                box_agent.update_q_table(state["box"], box_action, reward, next_state["box"])
                box_agent.replay()
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 更新探索率
            node_agent.update_epsilon()
            box_agent.update_epsilon()

        # 如果当前回合表现更好，更新最佳记录
        print(f"Total reward: {total_reward}, Best reward: {node_agent.best_reward}")
        if total_reward > node_agent.best_reward:
            node_agent.best_reward = total_reward
            box_agent.best_reward = total_reward
            node_agent.best_memory = deque(node_agent.memory, maxlen=node_agent.memory.maxlen)
            box_agent.best_memory = deque(box_agent.memory, maxlen=box_agent.memory.maxlen)
            node_agent.replay_best()
            box_agent.replay_best()

        # 保存训练结果
        rewards.append(total_reward)
        save_multi_agent_results(node_agent, box_agent, episode, total_reward, output_path, selected_prefix)

        # 计算最终评估指标
        mean_feature_pos = np.mean(list(nx.get_edge_attributes(multi_env.node_env.G, 'feature_pos').values()))
        mean_feature_cross = np.mean(list(nx.get_edge_attributes(multi_env.node_env.G, 'feature_cross').values()))
        pos_ratio_in_box = multi_env.box_env.calculate_pos_ratio_in_box()
        neg_ratio_out_box = multi_env.box_env.calculate_neg_ratio_out_box()
        feature_distance = multi_env.box_env.calculate_feature_distance()
        
        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}")
        print(f"Node metrics - Final pos: {mean_feature_pos}, Final cross: {mean_feature_cross}")
        print(f"Box metrics - Pos ratio in box: {pos_ratio_in_box}, Neg ratio out box: {neg_ratio_out_box}, Feature distance: {feature_distance}")

        # 计算并显示时间信息
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * (episodes / (episode + 1))
        remaining_time = estimated_total_time - elapsed_time
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, Estimated Total Time: {timedelta(seconds=int(estimated_total_time))}, Remaining Time: {timedelta(seconds=int(remaining_time))}")

        # 更新并保存最佳模型
        if (mean_feature_pos < node_agent.best_pos and 
            mean_feature_cross > node_agent.best_cross and
            pos_ratio_in_box > 0.6 and
            neg_ratio_out_box > 0.6):
            print('Update best model!')
            node_agent.best_pos = mean_feature_pos
            node_agent.best_cross = mean_feature_cross
            node_agent.best_q_table = node_agent.q_table.copy()
            box_agent.best_q_table = box_agent.q_table.copy()
            save_multi_agent_best_q_table(node_agent, box_agent, output_path)

        # 保存不同指标下的最佳模型
        if total_reward > node_agent.best_reward_save:
            node_agent.best_reward_save = total_reward
            box_agent.best_reward_save = total_reward
            save_multi_agent_best_model(node_agent, box_agent, output_path, 'best_reward_model.pkl')

        if mean_feature_pos < node_agent.best_pos_feature_distance:
            node_agent.best_pos_feature_distance = mean_feature_pos
            save_best_model(node_agent, output_path, 'best_pos_feature_distance_model.pkl')

        if mean_feature_cross > node_agent.best_cross_feature_distance:
            node_agent.best_cross_feature_distance = mean_feature_cross
            save_best_model(node_agent, output_path, 'best_cross_feature_distance_model.pkl')

        # 输出最终节点统计信息
        final_pos_count = len(multi_env.node_env.pos_nodes)
        final_neg_count = len(multi_env.node_env.neg_nodes)
        final_bbox = multi_env.box_env.bbox
        print(f"Episode {episode + 1}: Final positive nodes count: {final_pos_count}, Final negative nodes count: {final_neg_count}")
        print(f"Final bounding box: {final_bbox}")

    return rewards

def save_multi_agent_results(node_agent, box_agent, episode, reward, output_path, prefix):
    """保存多智能体训练结果"""
    node_state = node_agent.env.get_state()
    box_state, bbox = box_agent.env.get_state()
    
    pos_nodes = [node for node, data in node_state.nodes(data=True) if data['category'] == 'pos']
    neg_nodes = [node for node, data in node_state.nodes(data=True) if data['category'] == 'neg']

    with open(f"{output_path}/{prefix}_rewards.txt", "a") as f:
        f.write(f"Episode {episode}: Reward: {reward}\n")
        f.write(f"Positive nodes: {len(pos_nodes)}, Negative nodes: {len(neg_nodes)}\n")
        f.write(f"Bounding box: {bbox}\n\n")

def save_multi_agent_best_q_table(node_agent, box_agent, output_path):
    """保存最佳Q表"""
    with open(f"{output_path}/node_best_q_table.pkl", "wb") as f:
        torch.save(node_agent.best_q_table, f)
    with open(f"{output_path}/box_best_q_table.pkl", "wb") as f:
        torch.save(box_agent.best_q_table, f)

def save_multi_agent_best_model(node_agent, box_agent, output_path, filename):
    """保存多智能体最佳模型"""
    with open(f"{output_path}/node_{filename}", "wb") as f:
        torch.save(node_agent.q_table, f)
    with open(f"{output_path}/box_{filename}", "wb") as f:
        torch.save(box_agent.q_table, f)
    print(f"Best multi-agent model saved with reward: {node_agent.best_reward}")

def save_best_q_table(agent, output_path):
    """Save the best Q-table."""
    with open(f"{output_path}/best_q_table.pkl", "wb") as f:
        torch.save(agent.best_q_table, f)

def save_best_model(agent, output_path, filename):
    """Save the best model based on the highest reward."""
    with open(f"{output_path}/{filename}", "wb") as f:
        torch.save(agent.q_table, f)
    print(f"Best model saved with reward: {agent.best_reward}")

def main():
    """Main function to train the multi-agent system."""
    # 设置初始提示数据目录路径
    base_dir = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY, 'initial_prompts')
    # 获取目录下所有文件
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    # 提取文件名前缀（前三个下划线分隔的部分）
    file_prefixes = set('_'.join(f.split('_')[:3]) for f in files)
    # 训练参数设置
    max_steps = 100 # 每个episode的最大步数
    
    # 初始化节点智能体和矩形框智能体
    node_env = NodeOptimizationEnv
    box_env = BoxOptimizationEnv
    node_agent = NodeAgent(node_env)
    box_agent = BoxAgent(box_env)
    
    # 创建输出目录
    current_time = datetime.now().strftime("%Y%m%d_%H%M") # 当前时间
    output_path = os.path.join(os.path.dirname(__file__), 'train', current_time) # 输出目录
    
    # 训练多智能体系统
    rewards = train_multi_agent(
        node_agent, 
        box_agent, 
        episodes=30, 
        output_path=output_path, 
        base_dir=base_dir, 
        file_prefixes=file_prefixes, 
        max_steps=max_steps
    )
    
    # 绘制训练奖励曲线
    import matplotlib.pyplot as plt
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Multi-Agent Training Rewards Over Episodes')
    plt.savefig(f"{output_path}/rewards_plot.png")
    plt.show()

if __name__ == "__main__":
    main()
