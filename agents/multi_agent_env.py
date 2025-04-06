from agents.node_env import NodeOptimizationEnv
from agents.box_env import BoxOptimizationEnv
from utils import calculate_center_points

class MultiAgentEnv:
    def __init__(self, G, bbox_initial=None, image_size=560, max_steps=100, step_size=14):
        self.node_env = NodeOptimizationEnv(G, max_steps)
        self.box_env = BoxOptimizationEnv(G, bbox_initial, image_size, max_steps, step_size)
        self.node_env.boxes = self.box_env.boxes  # 初始化时共享boxes
        self.steps = 0
        self.max_steps = max_steps
        self.features = None  # 添加特征存储
        self.best_reward = -float('inf')  # 初始化最佳奖励
        
    def reset(self):
        """重置环境"""
        node_state = self.node_env.reset()
        box_state = self.box_env.reset()
        self.steps = 0
        
        # 确保特征在重置时也被正确设置
        if hasattr(self.node_env, 'features'):
            self.features = self.node_env.features
            self.box_env.features = self.features
            
        return {"node": node_state, "box": box_state}
    
    def step(self, action_dict): 
        # 执行节点智能体的动作
        if "node" in action_dict:
            node_next_state, node_reward, node_done = self.node_env.step(action_dict["node"])
            # 同步图结构和特征到box环境
            self.box_env.G = self.node_env.G.copy()
            self.node_env.boxes = self.box_env.boxes  # 保持boxes同步
            if hasattr(self.node_env, 'features'):
                self.features = self.node_env.features
                self.box_env.features = self.features
        else:
            node_next_state, node_reward, node_done = self.node_env.get_state(), 0, False
        
        # 执行边界框智能体的动作
        if "box" in action_dict:
            box_next_state, box_reward, box_done = self.box_env.step(action_dict["box"])
        else:
            box_next_state, box_reward, box_done = self.box_env.get_state(), 0, False
        
        # 更新步数
        self.steps += 1
        
        # 组合奖励
        reward = node_reward + box_reward
        done = node_done or box_done or self.steps >= self.max_steps

        # 检查是否达到最大步数
        if self.steps >= self.max_steps:
            # 移除所有box内的负提示点
            neg_nodes_to_remove = []
            for node in self.node_env.G.nodes():
                if self.node_env.G.nodes[node]['category'] == 'neg':
                    point = calculate_center_points([node], self.box_env.image_size)[0]
                    # 检查点是否在任意box内
                    for box in self.box_env.boxes:
                        if (box['min_x'] <= point[0] <= box['max_x'] and 
                            box['min_y'] <= point[1] <= box['max_y']):
                            neg_nodes_to_remove.append(node)
                            break

            # 移除所有box外的正提示点
            pos_nodes_to_remove = []
            for node in self.node_env.G.nodes():
                if self.node_env.G.nodes[node]['category'] == 'pos':
                    point = calculate_center_points([node], self.box_env.image_size)[0]
                    # 检查点是否在所有box外部
                    outside_all_boxes = True
                    for box in self.box_env.boxes:
                        if (box['min_x'] <= point[0] <= box['max_x'] and
                            box['min_y'] <= point[1] <= box['max_y']):
                            outside_all_boxes = False
                            break
                    if outside_all_boxes:
                        pos_nodes_to_remove.append(node)

            # 移除这些节点
            for node in neg_nodes_to_remove + pos_nodes_to_remove:
                if node in self.node_env.G.nodes():
                    if self.node_env.G.nodes[node]['category'] == 'neg':
                        self.node_env.remove_node(node, "neg")
                    else:
                        self.node_env.remove_node(node, "pos")
                    # 确保box环境的图也保持同步
                    if node in self.box_env.G.nodes():
                        self.box_env.G.remove_node(node)
            
            # 在移除节点后更新状态
            node_next_state = self.node_env.get_state()
            box_next_state = self.box_env.get_state()
        
        return {"node": node_next_state, "box": box_next_state}, reward, done
    
    def get_state(self):
        """获取当前状态"""
        return {"node": self.node_env.get_state(), "box": self.box_env.get_state()} 