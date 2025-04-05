import os
import torch
import numpy as np
from collections import defaultdict, deque
import random
import networkx as nx
from datetime import timedelta
import time
from agents import NodeOptimizationEnv, NodeAgent, BoxAgent, MultiAgentEnv, BoxOptimizationEnv
from utils import calculate_distances, convert_to_edges
from tqdm import tqdm
import json


def test_multi_agent(node_agent, box_agent, test_base_dir, test_file_prefixes, image_size, device, output_path):
    """Test the multi-agent system on a set of test data."""
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()

    for prefix in test_file_prefixes:
        print(f"Testing with file prefix: {prefix}")

        feature_file = os.path.join(test_base_dir, f"{prefix}_features.pt")
        pos_file = os.path.join(test_base_dir, f"{prefix}_initial_indices_pos.pt")
        neg_file = os.path.join(test_base_dir, f"{prefix}_initial_indices_neg.pt")
        bbox_file = os.path.join(test_base_dir, f"{prefix}_bbox.json")

        if not (os.path.exists(feature_file) and os.path.exists(pos_file) and os.path.exists(neg_file) and os.path.exists(bbox_file)):
            print(f"Required files not found for prefix {prefix}, skipping this test.")
            continue

        features = torch.load(feature_file).to(device)
        positive_indices = torch.load(pos_file).to(device)
        negative_indices = torch.load(neg_file).to(device)
        
        with open(bbox_file, 'r') as f:
            bbox_data = json.load(f)

        positive_indices = torch.unique(positive_indices).to(device)
        negative_indices = torch.unique(negative_indices).to(device)

        # Remove intersections between positive and negative indices
        intersection = set(positive_indices.tolist()).intersection(set(negative_indices.tolist()))
        positive_indices = torch.tensor([x for x in positive_indices.cpu().tolist() if x not in intersection]).cuda()
        negative_indices = torch.tensor([x for x in negative_indices.cpu().tolist() if x not in intersection]).cuda()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
            features, positive_indices, negative_indices, image_size, device)

        feature_pos_edge = convert_to_edges(positive_indices, positive_indices, feature_pos_distances)
        physical_pos_edge = convert_to_edges(positive_indices, positive_indices, physical_pos_distances)
        physical_neg_edge = convert_to_edges(negative_indices, negative_indices, physical_neg_distances)
        feature_cross_edge = convert_to_edges(positive_indices, negative_indices, feature_cross_distances)
        physical_cross_edge = convert_to_edges(positive_indices, negative_indices, physical_cross_distances)

        G = nx.MultiGraph()
        G.add_nodes_from(positive_indices.cpu().numpy(), category='pos')
        G.add_nodes_from(negative_indices.cpu().numpy(), category='neg')

        G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
        G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
        G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
        G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
        G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

        # Initialize multi-agent environment
        multi_env = MultiAgentEnv(G, bbox_data, image_size, max_steps=100)
        node_agent.env = multi_env.node_env
        box_agent.env = multi_env.box_env
        
        state = multi_env.reset()
        done = False
        total_reward = 0
        step_count = 0

        while not done:
            if step_count % 2 == 0:  # 偶数步执行节点智能体动作
                node_action = node_agent.get_action(state["node"])
                action_dict = {"node": node_action}
            else:  # 奇数步执行矩形框智能体动作
                box_action = box_agent.get_action(state["box"])
                action_dict = {"box": box_action}
                
            next_state, reward, done = multi_env.step(action_dict)
            state = next_state
            total_reward += reward
            step_count += 1

        # Save the final state
        final_pos_nodes = [node for node in multi_env.node_env.pos_nodes]
        final_neg_nodes = [node for node in multi_env.node_env.neg_nodes]
        final_bbox = multi_env.box_env.bbox

        torch.save(final_pos_nodes, os.path.join(output_path, f"{prefix}_optimized_pos.pt"))
        torch.save(final_neg_nodes, os.path.join(output_path, f"{prefix}_optimized_neg.pt"))
        
        # Save final bounding box
        with open(os.path.join(output_path, f"{prefix}_optimized_bbox.json"), 'w') as f:
            json.dump(final_bbox, f)

        elapsed_time = time.time() - start_time
        print(f"Test with prefix {prefix} completed, Total Reward: {total_reward}")
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}")


def optimize_with_multi_agent(node_agent, box_agent, pos_indices, neg_indices, features, bbox_data, max_steps, device, image_size):
    """Optimize node positions and bounding box using the multi-agent system."""
    pos_indices = torch.unique(pos_indices).to(device)
    neg_indices = torch.unique(neg_indices).to(device)

    # Remove intersections between positive and negative indices
    intersection = set(pos_indices.tolist()).intersection(set(neg_indices.tolist()))
    pos_indices = torch.tensor([x for x in pos_indices.cpu().tolist() if x not in intersection]).cuda()
    neg_indices = torch.tensor([x for x in neg_indices.cpu().tolist() if x not in intersection]).cuda()
    
    if pos_indices.numel() == 0 or neg_indices.numel() == 0:
        return pos_indices, neg_indices, bbox_data

    feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
        features, pos_indices, neg_indices, image_size, device)

    feature_pos_edge = convert_to_edges(pos_indices, pos_indices, feature_pos_distances)
    physical_pos_edge = convert_to_edges(pos_indices, pos_indices, physical_pos_distances)
    physical_neg_edge = convert_to_edges(neg_indices, neg_indices, physical_neg_distances)
    feature_cross_edge = convert_to_edges(pos_indices, neg_indices, feature_cross_distances)
    physical_cross_edge = convert_to_edges(pos_indices, neg_indices, physical_cross_distances)

    G = nx.MultiGraph()
    G.add_nodes_from(pos_indices.cpu().numpy(), category='pos')
    G.add_nodes_from(neg_indices.cpu().numpy(), category='neg')

    G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
    G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
    G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
    G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
    G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

    multi_env = MultiAgentEnv(G, bbox_data, image_size, max_steps)
    node_agent.env = multi_env.node_env
    box_agent.env = multi_env.box_env
    
    state = multi_env.reset()
    done = False
    step_count = 0

    while not done:
        if step_count % 2 == 0:  # 偶数步执行节点智能体动作
            node_action = node_agent.get_action(state["node"])
            action_dict = {"node": node_action}
        else:  # 奇数步执行矩形框智能体动作
            box_action = box_agent.get_action(state["box"])
            action_dict = {"box": box_action}
            
        next_state, _, done = multi_env.step(action_dict)
        state = next_state
        step_count += 1

    optimized_pos_indices = torch.tensor([node for node in multi_env.node_env.pos_nodes])
    optimized_neg_indices = torch.tensor([node for node in multi_env.node_env.neg_nodes])
    optimized_bbox = multi_env.box_env.bbox

    return optimized_pos_indices, optimized_neg_indices, optimized_bbox


def test_agent(agent, test_base_dir, test_file_prefixes, image_size, device, output_path):
    """Test the Q-learning agent on a set of test data."""
    os.makedirs(output_path, exist_ok=True)
    start_time = time.time()

    for prefix in test_file_prefixes:
        print(f"Testing with file prefix: {prefix}")

        feature_file = os.path.join(test_base_dir, f"{prefix}_features.pt")
        pos_file = os.path.join(test_base_dir, f"{prefix}_initial_indices_pos.pt")
        neg_file = os.path.join(test_base_dir, f"{prefix}_initial_indices_neg.pt")

        if not (os.path.exists(feature_file) and os.path.exists(pos_file) and os.path.exists(neg_file)):
            print(f"Required files not found for prefix {prefix}, skipping this test.")
            continue

        features = torch.load(feature_file).to(device)
        positive_indices = torch.load(pos_file).to(device)
        negative_indices = torch.load(neg_file).to(device)

        positive_indices = torch.unique(positive_indices).to(device)
        negative_indices = torch.unique(negative_indices).to(device)

        # Remove intersections between positive and negative indices
        intersection = set(positive_indices.tolist()).intersection(set(negative_indices.tolist()))
        positive_indices = torch.tensor([x for x in positive_indices.cpu().tolist() if x not in intersection]).cuda()
        negative_indices = torch.tensor([x for x in negative_indices.cpu().tolist() if x not in intersection]).cuda()
        if positive_indices.numel() == 0 or negative_indices.numel() == 0:
            continue

        feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
            features, positive_indices, negative_indices, image_size, device)

        feature_pos_edge = convert_to_edges(positive_indices, positive_indices, feature_pos_distances)
        physical_pos_edge = convert_to_edges(positive_indices, positive_indices, physical_pos_distances)
        physical_neg_edge = convert_to_edges(negative_indices, negative_indices, physical_neg_distances)
        feature_cross_edge = convert_to_edges(positive_indices, negative_indices, feature_cross_distances)
        physical_cross_edge = convert_to_edges(positive_indices, negative_indices, physical_cross_distances)

        G = nx.MultiGraph()
        G.add_nodes_from(positive_indices.cpu().numpy(), category='pos')
        G.add_nodes_from(negative_indices.cpu().numpy(), category='neg')

        G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
        G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
        G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
        G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
        G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

        agent.env = NodeOptimizationEnv(G, max_steps=100)
        state = agent.env.reset()
        done = False
        total_reward = 0

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = agent.env.step(action)
            state = next_state
            total_reward += reward

        # Save the final state graph with optimized pos and neg nodes
        final_G = agent.env.get_state()
        pos_nodes = [node for node, data in final_G.nodes(data=True) if data['category'] == 'pos']
        neg_nodes = [node for node, data in final_G.nodes(data=True) if data['category'] == 'neg']

        torch.save(pos_nodes, os.path.join(output_path, f"{prefix}_optimized_pos.pt"))
        torch.save(neg_nodes, os.path.join(output_path, f"{prefix}_optimized_neg.pt"))

        elapsed_time = time.time() - start_time
        print(f"Test with prefix {prefix} completed, Total Reward: {total_reward}")
        print(f"Elapsed Time: {timedelta(seconds=int(elapsed_time))}")


def optimize_nodes(agent, pos_indices, neg_indices, features, max_steps, device, image_size):
    """Optimize node positions using the Q-learning agent."""
    pos_indices = torch.unique(pos_indices).to(device)
    neg_indices = torch.unique(neg_indices).to(device)

    # Remove intersections between positive and negative indices
    intersection = set(pos_indices.tolist()).intersection(set(neg_indices.tolist()))
    pos_indices = torch.tensor([x for x in pos_indices.cpu().tolist() if x not in intersection]).cuda()
    neg_indices = torch.tensor([x for x in neg_indices.cpu().tolist() if x not in intersection]).cuda()
    if pos_indices.numel() == 0 or neg_indices.numel() == 0:
        return pos_indices, neg_indices

    feature_pos_distances, feature_cross_distances, physical_pos_distances, physical_neg_distances, physical_cross_distances = calculate_distances(
        features, pos_indices, neg_indices, image_size, device)

    feature_pos_edge = convert_to_edges(pos_indices, pos_indices, feature_pos_distances)
    physical_pos_edge = convert_to_edges(pos_indices, pos_indices, physical_pos_distances)
    physical_neg_edge = convert_to_edges(neg_indices, neg_indices, physical_neg_distances)
    feature_cross_edge = convert_to_edges(pos_indices, neg_indices, feature_cross_distances)
    physical_cross_edge = convert_to_edges(pos_indices, neg_indices, physical_cross_distances)

    G = nx.MultiGraph()
    G.add_nodes_from(pos_indices.cpu().numpy(), category='pos')
    G.add_nodes_from(neg_indices.cpu().numpy(), category='neg')

    G.add_weighted_edges_from(feature_pos_edge, weight='feature_pos')
    G.add_weighted_edges_from(physical_pos_edge, weight='physical_pos')
    G.add_weighted_edges_from(physical_neg_edge, weight='physical_neg')
    G.add_weighted_edges_from(feature_cross_edge, weight='feature_cross')
    G.add_weighted_edges_from(physical_cross_edge, weight='physical_cross')

    agent.env = NodeOptimizationEnv(G, max_steps)
    state = agent.env.reset()
    done = False
    while not done:
        action = agent.get_action(state)
        next_state, reward, done = agent.env.step(action)
        state = next_state

    optimized_pos_indices = torch.tensor([node for node in agent.env.pos_nodes])
    optimized_neg_indices = torch.tensor([node for node in agent.env.neg_nodes])
    return optimized_pos_indices, optimized_neg_indices
