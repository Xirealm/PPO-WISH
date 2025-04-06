import os
import torch
import numpy as np
import networkx as nx
import random
from collections import deque
import time
from datetime import timedelta, datetime
from agents import NodeOptimizationEnv, NodeAgent, BoxAgent, MultiAgentEnv, BoxOptimizationEnv
from utils import calculate_distances, convert_to_edges, get_box_node_indices
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
    best_performance = {
        'reward': float('-inf'),
        'pos_feature_sim': float('inf'),
        'cross_feature_sim': float('-inf'),
        'pos_ratio_in_box': 0,
        'neg_ratio_out_box': 0
    }
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
            boxes = bbox_data['cluster_boxes']  # 使用聚类得到的多个box
            
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
        inside_neg_indices, outside_neg_indices) = get_box_node_indices(G, boxes)
        
        print(f"Inside bbox - Pos: {len(inside_pos_indices)}, Neg: {len(inside_neg_indices)}")
        print(f"Outside bbox - Pos: {len(outside_pos_indices)}, Neg: {len(outside_neg_indices)}")

        # 初始化多智能体环境
        multi_env = MultiAgentEnv(G, boxes, image_size, max_steps)
        node_agent.env = multi_env.node_env
        box_agent.env = multi_env.box_env
        
        # 设置特征信息到两个环境中
        multi_env.node_env.features = features
        multi_env.box_env.features = features
        
        state = multi_env.reset()
        done = False
        total_reward = 0

        # 根据最佳奖励动态调整epsilon
        normalized_reward = (node_agent.best_reward - 0) / (5 - 0)
        if node_agent.best_reward < 0:
            node_agent.epsilon = node_agent.epsilon_start
            box_agent.epsilon = box_agent.epsilon_start
        elif node_agent.best_reward >= 10:
            node_agent.epsilon = node_agent.epsilon_end
            box_agent.epsilon = box_agent.epsilon_end
        else:
            node_agent.epsilon = 1 - normalized_reward
            box_agent.epsilon = 1 - normalized_reward
        print(f"best_reward:{node_agent.best_reward},node_epsilon:{node_agent.epsilon},box_epsilon:{box_agent.epsilon}")

        # 训练循环 - 交替执行节点智能体和矩形框智能体的动作  
        step_count = 0
        while not done:
            # 在每个时间步,决定要执行的智能体
            if step_count % 2 == 0:  # 偶数步执行节点智能体动作
                state_tensor = node_agent._get_state_features(state["node"])
                node_action = node_agent.get_action(state["node"])
                action_dict = {"node": node_action}
                next_state, reward, done = multi_env.step(action_dict)
                
                # 更新节点智能体记忆和网络
                next_state_tensor = node_agent._get_state_features(next_state["node"])
                node_agent.memory.append((state_tensor, node_action, reward, next_state_tensor))
                node_agent.replay()
            else:  # 奇数步执行矩形框智能体动作
                state_tensor = box_agent._get_state_features(state["box"])
                box_action = box_agent.get_action(state["box"])
                action_dict = {"box": box_action}
                next_state, reward, done = multi_env.step(action_dict)
                
                # 更新矩形框智能体记忆和网络  
                next_state_tensor = box_agent._get_state_features(next_state["box"])
                box_agent.memory.append((state_tensor, box_action, reward, next_state_tensor))
                box_agent.replay()
            
            state = next_state
            total_reward += reward
            step_count += 1
            
            # 更新目标网络
            if step_count % 100 == 0:
                node_agent.update_target_network()
                box_agent.update_target_network()

        # 如果当前回合表现更好，更新最佳记录和保存最大reward模型
        print(f"Total reward: {total_reward}, Best reward: {node_agent.best_reward}")
        if total_reward > node_agent.best_reward:
            node_agent.best_reward = total_reward
            box_agent.best_reward = total_reward
            node_agent.best_memory = deque(node_agent.memory, maxlen=node_agent.memory.maxlen)
            box_agent.best_memory = deque(box_agent.memory, maxlen=box_agent.memory.maxlen)
            node_agent.replay_best()
            box_agent.replay_best()

            # 保存最佳模型到output_path
            model_path = os.path.join(output_path, 'best_models')
            os.makedirs(model_path, exist_ok=True)
            
            # 保存完整模型状态
            torch.save({
                'node_policy_net': node_agent.policy_net.state_dict(),
                'box_policy_net': box_agent.policy_net.state_dict(),
                'node_target_net': node_agent.target_net.state_dict(),
                'box_target_net': box_agent.target_net.state_dict(),
                'node_optimizer': node_agent.optimizer.state_dict(),
                'box_optimizer': box_agent.optimizer.state_dict(),
                'performance_metrics': best_performance,
                'episode': episode
            }, os.path.join(model_path, 'best_model.pt'))
            
            # 保存单独的模型文件到output_path的model目录
            output_model_dir = os.path.join(output_path, 'model')
            os.makedirs(output_model_dir, exist_ok=True)
            torch.save(node_agent.policy_net.state_dict(), os.path.join(output_model_dir, 'node_best_model.pkl'))
            torch.save(box_agent.policy_net.state_dict(), os.path.join(output_model_dir, 'box_best_model.pkl'))
            
            print(f'Updated best model at episode {episode}')
            print(f'Performance metrics: {best_performance}')
            
            # 保存性能指标到output_path
            with open(os.path.join(output_model_dir, 'performance_metrics.json'), 'w') as f:
                json.dump(best_performance, f, indent=4)

        # 保存训练结果
        rewards.append(total_reward)
        save_multi_agent_results(node_agent, box_agent, episode, total_reward, output_path, selected_prefix)

        # 计算并显示时间信息
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time * (episodes / (episode + 1))
        remaining_time = estimated_total_time - elapsed_time
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}, Estimated Total Time: {timedelta(seconds=int(estimated_total_time))}, Remaining Time: {timedelta(seconds=int(remaining_time))}")

        # 输出最终节点统计信息
        final_pos_count = len(multi_env.node_env.pos_nodes)
        final_neg_count = len(multi_env.node_env.neg_nodes)
        final_boxes = multi_env.box_env.boxes
        print(f"Episode {episode + 1}: Final positive nodes count: {final_pos_count}, Final negative nodes count: {final_neg_count}")
        print(f"Final boxes count: {len(final_boxes)}")
        for i, box in enumerate(final_boxes):
            print(f"Final box {i}: {box}")

    return rewards, best_performance

def save_multi_agent_results(node_agent, box_agent, episode, reward, output_path, prefix):
    """保存多智能体训练结果"""
    node_state = node_agent.env.get_state()
    box_state, boxes = box_agent.env.get_state()
    
    pos_nodes = [node for node, data in node_state.nodes(data=True) if data['category'] == 'pos']
    neg_nodes = [node for node, data in node_state.nodes(data=True) if data['category'] == 'neg']

    # 保存每个episode的训练结果
    episode_dir = os.path.join(output_path, 'episode_results')
    os.makedirs(episode_dir, exist_ok=True)
    result_path = os.path.join(episode_dir, f'episode_{episode}')
    os.makedirs(result_path, exist_ok=True)

    # 保存模型状态
    torch.save(node_agent.policy_net.state_dict(), 
              os.path.join(result_path, f'node_policy_net_ep{episode}.pkl'))
    torch.save(box_agent.policy_net.state_dict(), 
              os.path.join(result_path, f'box_policy_net_ep{episode}.pkl'))

    # 保存训练信息
    with open(os.path.join(result_path, f"{prefix}_info.txt"), "w") as f:
        f.write(f"Episode {episode}: Reward: {reward}\n")
        f.write(f"Positive nodes: {len(pos_nodes)}, Negative nodes: {len(neg_nodes)}\n")
        f.write(f"Number of boxes: {len(boxes)}\n")
        for i, box in enumerate(boxes):
            f.write(f"Box {i}: {box}\n")
        f.write("\n")

def main():
    """Main function to train the multi-agent system."""
    # 设置初始提示数据目录路径
    base_dir = os.path.join(BASE_DIR, 'results', DATASET, CATAGORY, 'initial_prompts')
    # 获取目录下所有文件
    files = [f for f in os.listdir(base_dir) if os.path.isfile(os.path.join(base_dir, f))]
    # 提取文件名前缀（前三个下划线分隔的部分）
    file_prefixes = set('_'.join(f.split('_')[:3]) for f in files)
    # 训练参数设置
    episodes = 10  # 训练回合数
    max_steps = 100  # 每个episode的最大步数
    
    # 创建输出目录结构
    current_time = datetime.now().strftime("%Y%m%d_%H%M")
    output_path = os.path.join(os.path.dirname(__file__), 'train', current_time)
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.join(output_path, 'episode_results'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'best_models'), exist_ok=True)
    os.makedirs(os.path.join(output_path, 'model'), exist_ok=True)
    
    # 保存训练配置
    config = {
        'episodes': episodes,
        'max_steps': max_steps,
        'dataset': DATASET,
        'category': CATAGORY,
        'image_size': SIZE
    }
    with open(os.path.join(output_path, 'config.json'), 'w') as f:
        json.dump(config, f)
    
    # 初始化节点智能体和矩形框智能体
    node_env = NodeOptimizationEnv
    box_env = BoxOptimizationEnv
    node_agent = NodeAgent(node_env)
    box_agent = BoxAgent(box_env)
    
    # 尝试加载已有模型，如果存在的话
    try:
        node_model_path = os.path.join(output_path, 'model', 'node_best_model.pkl')
        box_model_path = os.path.join(output_path, 'model', 'box_best_model.pkl')
        
        if os.path.exists(node_model_path) and os.path.exists(box_model_path):
            print("加载已有的模型继续训练...")
            node_agent.policy_net.load_state_dict(torch.load(node_model_path))
            node_agent.target_net.load_state_dict(torch.load(node_model_path))
            box_agent.policy_net.load_state_dict(torch.load(box_model_path))
            box_agent.target_net.load_state_dict(torch.load(box_model_path))
            
            # 加载性能指标
            if os.path.exists(os.path.join(output_path, 'model', 'performance_metrics.json')):
                with open(os.path.join(output_path, 'model', 'performance_metrics.json'), 'r') as f:
                    metrics = json.load(f)
                    node_agent.best_reward = metrics.get('reward', node_agent.best_reward)
                    box_agent.best_reward = metrics.get('reward', box_agent.best_reward)
    except Exception as e:
        print(f"加载模型失败: {e}，将使用新模型开始训练")
    
    # 训练多智能体系统
    rewards, best_performance = train_multi_agent(
        node_agent, 
        box_agent, 
        episodes=episodes,
        output_path=output_path, 
        base_dir=base_dir, 
        file_prefixes=file_prefixes, 
        max_steps=max_steps
    )
    
    # 保存训练结果，包括最佳性能信息
    training_info = {
        'rewards': rewards,
        'best_performance': best_performance,
        'timestamps': [str(datetime.now())]
    }
    torch.save(training_info, os.path.join(output_path, 'training_info.pt'))
    
    # 保存最终模型状态到output_path
    torch.save({
        'node_policy_net': node_agent.policy_net.state_dict(),
        'box_policy_net': box_agent.policy_net.state_dict(),
        'node_target_net': node_agent.target_net.state_dict(),
        'box_target_net': box_agent.target_net.state_dict(),
        'node_optimizer': node_agent.optimizer.state_dict(),
        'box_optimizer': box_agent.optimizer.state_dict(),
        'performance_metrics': best_performance,
        'episode': episodes - 1
    }, os.path.join(output_path, 'final_model.pt'))
    
    # 保存最终模型的单文件版本到output_path
    output_model_dir = os.path.join(output_path, 'model')
    os.makedirs(output_model_dir, exist_ok=True)
    torch.save(node_agent.policy_net.state_dict(), os.path.join(output_model_dir, 'node_final_model.pkl'))
    torch.save(box_agent.policy_net.state_dict(), os.path.join(output_model_dir, 'box_final_model.pkl'))
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Multi-Agent Training Rewards Over Episodes')
    plt.savefig(os.path.join(output_path, "rewards_plot.png"))
    plt.close()

if __name__ == "__main__":
    main()
