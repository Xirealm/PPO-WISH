import random
from collections import defaultdict, deque
from agents.box_env import BoxOptimizationEnv

class BoxAgent:
    def __init__(self, env:BoxOptimizationEnv, alpha=0.1, gamma=0.9, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.995):
        """初始化box智能体"""
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start  # 存储初始epsilon值
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.q_table = defaultdict(float)
        
        # 添加其他必要的属性
        self.best_reward = -float('inf')  # 用于记录最佳奖励
        self.best_reward_save = -float('inf')
        self.memory = deque(maxlen=10000)  # 添加经验回放内存
        self.best_memory = deque(maxlen=10000)  # 添加最佳经验内存
        self.batch_size = 64  # 添加批量大小
        
        # 动作空间设计
        self.edge_actions = ["shrink_up", "shrink_down", "shrink_left", "shrink_right",
                             "expand_up", "expand_down", "expand_left", "expand_right"]
        self.merge_action = ["merge"]
        self.actions = self.edge_actions + self.merge_action

    def update_epsilon(self):
        """更新探索率"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def get_box_type(self, box_idx):
        """判断box类型:是否为边界box,是否四面均有邻接"""
        box = self.env.boxes[box_idx]
        
        # 判断是否是边界box(是否具有极值坐标)
        min_x = min(b['min_x'] for b in self.env.boxes)
        max_x = max(b['max_x'] for b in self.env.boxes)
        min_y = min(b['min_y'] for b in self.env.boxes)
        max_y = max(b['max_y'] for b in self.env.boxes)
        
        is_extreme = (box['min_x'] == min_x or box['max_x'] == max_x or 
                     box['min_y'] == min_y or box['max_y'] == max_y)
        
        # 检查四个方向的邻接情况
        free_edges = self.env.check_box_adjacency(box_idx)
        all_adjacent = not any(free_edges.values())  # 四面均有邻接
        
        return is_extreme, all_adjacent, free_edges

    def get_action(self, state):
        """根据box类型选择合适的动作"""
        if random.random() < self.epsilon:
            # 50%概率选择极值box,50%概率选择四面邻接box
            operation_type = random.choice(['extreme', 'adjacent'])
            extreme_boxes = []
            adjacent_boxes = []
            
            # 分类所有box
            for box_idx in range(len(self.env.boxes)):
                is_extreme, all_adjacent, free_edges = self.get_box_type(box_idx)
                if is_extreme:
                    extreme_boxes.append((box_idx, free_edges))
                if all_adjacent:
                    adjacent_boxes.append(box_idx)
            
            if operation_type == 'extreme' and extreme_boxes:
                # 随机选择一个极值box
                box_idx, free_edges = random.choice(extreme_boxes)
                # 在可用的方向中随机选择一个
                available_directions = [d for d, is_free in free_edges.items() if is_free]
                if available_directions:
                    direction = random.choice(available_directions)
                    operation = random.choice([f"shrink_{direction}", f"expand_{direction}"])
                    return (box_idx, operation, None)
                    
            elif operation_type == 'adjacent' and adjacent_boxes:
                # 随机选择一个四面邻接的box
                box_idx = random.choice(adjacent_boxes)
                mergeable = self.env.find_mergeable_boxes(box_idx)
                if mergeable:
                    target_idx, _ = mergeable[0]
                    return (box_idx, "merge", target_idx)
                    
            # 如果所选类型无可用动作,随机选择一个box和动作
            box_idx = random.randint(0, len(self.env.boxes) - 1)
            return (box_idx, random.choice(self.actions), None)
            
        else:
            # 使用Q表选择动作
            state_hash = self._get_state_hash(state)
            best_value = float('-inf')
            best_action = None
            
            # 同样遵循50%的概率分配
            if random.random() < 0.5:
                # 评估极值box的动作
                for box_idx in range(len(self.env.boxes)):
                    is_extreme, _, free_edges = self.get_box_type(box_idx)
                    if is_extreme:
                        for direction, is_free in free_edges.items():
                            if is_free:
                                for op in [f"shrink_{direction}", f"expand_{direction}"]:
                                    action = (box_idx, op, None)
                                    value = self.q_table.get((state_hash, action), 0)
                                    if value > best_value:
                                        best_value = value
                                        best_action = action
            else:
                # 评估四面邻接box的合并动作
                for box_idx in range(len(self.env.boxes)):
                    _, all_adjacent, _ = self.get_box_type(box_idx)
                    if all_adjacent:
                        mergeable = self.env.find_mergeable_boxes(box_idx)
                        for target_idx, similarity in mergeable:
                            action = (box_idx, "merge", target_idx)
                            # 在Q值基础上加入相似度作为额外考虑因素
                            value = self.q_table.get((state_hash, action), 0) + similarity
                            if value > best_value:
                                best_value = value
                                best_action = action
            
            return best_action if best_action else (0, "shrink_up", None)

    def _get_state_hash(self, state):
        """将状态转换为哈希值"""
        _, boxes = state
        box_tuples = tuple(
            (box['min_x'], box['min_y'], box['max_x'], box['max_y'])
            for box in boxes
        )
        return hash(box_tuples)

    def update_q_table(self, state, action, reward, next_state):
        """更新Q表,根据box类型选择动作空间"""
        state_hash = self._get_state_hash(state)
        next_state_hash = self._get_state_hash(next_state)
        
        # 获取next_state中每个box可用的动作
        next_actions = []
        for i in range(len(self.env.boxes)):
            is_extreme, all_adjacent, free_edges = self.get_box_type(i)
            if is_extreme:
                # 边界box的expand/shrink动作
                for direction, is_free in free_edges.items():
                    if is_free:
                        next_actions.extend([
                            (i, f"shrink_{direction}", None),
                            (i, f"expand_{direction}", None)
                        ])
            elif all_adjacent:
                # 四面邻接box的合并动作
                mergeable = self.env.find_mergeable_boxes(i)
                for target_idx, _ in mergeable:
                    next_actions.append((i, "merge", target_idx))
        
        # 计算最大Q值
        max_next_q = max(
            [self.q_table.get((next_state_hash, next_action), 0) 
             for next_action in next_actions],
            default=0)
        
        old_q = self.q_table.get((state_hash, action), 0)
        new_q = old_q + self.alpha * (reward + self.gamma * max_next_q - old_q)
        self.q_table[(state_hash, action)] = new_q

    def replay(self):
        """从记忆中回放经验"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state)

    def replay_best(self):
        """从最佳记忆中回放经验"""
        if len(self.best_memory) < self.batch_size:
            return
        
        batch = random.sample(self.best_memory, self.batch_size)
        for state, action, reward, next_state in batch:
            self.update_q_table(state, action, reward, next_state) 