#经验分布仿真（实验中）
import torch

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm
import numpy as np

#matplotlib.use('Qt5Agg') 

# 环境参数
GRID_SIZE = 200
OBSTACLE_NUM = 20       # 非线性障碍物数量
NUM_TARGET_TRAJ = 300  
TARGET_STEPS = 80      

# 遗传算法参数
POP_SIZE = 200
GEN_MAX = 300
CROSS_RATE = 0.8
MUTATE_RATE = 0.1
PATH_LENGTH = 800        

obstacle = [(3,6),(23,41),(61,64),(89,19),(193,132),(123,23),(112,63),(24,156),(34,63),(142,145),(132,23),(178,4),(99,181),(34,75)]


# 初始化环境（PyTorch实现）
class SearchEnvironment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.grid, self.target_prob = self.generate_environment()
        self.start_pos = torch.tensor([GRID_SIZE//2, GRID_SIZE//2])
        
    def generate_environment(self):
        # 生成非线性障碍物（连续区域）
        grid = torch.zeros((GRID_SIZE, GRID_SIZE), dtype=torch.float32)
        
        for i,j in obstacle:
            grid[i:i+10,j:j+10] = 1.0 
        
        # 生成目标概率分布
        target_prob = torch.rand((GRID_SIZE, GRID_SIZE))
        target_prob[grid.bool()] = 0.0  # 障碍物区域概率清零
        target_prob /= target_prob.sum()
        
        return grid, target_prob

# 蒙特卡洛目标轨迹生成（PyTorch加速）
class MonteCarloTarget(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.trajectories = []
        
    def generate_trajectories(self):
        print("Generating target trajectories...")
        # 批量生成轨迹（GPU加速）
        batch_size = 200
        for _ in tqdm(range(NUM_TARGET_TRAJ // batch_size)):
            traj_batch = self.generate_batch_trajectories(batch_size)
            self.trajectories.extend(traj_batch)
        self.calculate_probability()
        return self.trajectories
    
    def generate_batch_trajectories(self, batch_size):
        # 批量生成轨迹（优化GPU利用率）
        current_pos = self.random_start_batch(batch_size).float()
        traj = [current_pos]
        
        for _ in range(TARGET_STEPS):
            move = torch.randint(-1, 2, (batch_size, 2), dtype=torch.float32)
            new_pos = torch.clamp(current_pos + move, 0, GRID_SIZE-1)
            
            # 障碍物检测
            valid_mask = self.env.grid[new_pos[:,0].long(), new_pos[:,1].long()] == 0
            current_pos = torch.where(valid_mask.unsqueeze(1), new_pos, current_pos)
            traj.append(current_pos.clone())
        
        return [t.cpu().numpy() for t in torch.stack(traj).permute(1,0,2)]
    
    def random_start_batch(self, batch_size):
        # 批量随机起点生成
        starts = []
        while len(starts) < batch_size:
            candidates = torch.randint(0, GRID_SIZE, (batch_size*2, 2))
            valid = self.env.grid[candidates[:,0], candidates[:,1]] == 0
            starts.extend(candidates[valid][:batch_size-len(starts)])
        return torch.stack(starts[:batch_size])
    
    def calculate_probability(self):
        # 合并轨迹点并转换为浮点张量
        all_points = torch.cat([torch.tensor(traj, dtype=torch.float32) 
                            for traj in self.trajectories], dim=0)
        
        # 确保坐标在有效范围内 [0, GRID_SIZE-1]
        all_points = torch.clamp(all_points, 0, GRID_SIZE-1 - 1e-6)  # 避免浮点误差
        
        # 转换为网格索引（整数坐标）
        x_idx = (all_points[:, 0]).long()
        y_idx = (all_points[:, 1]).long()
        
        # 确保索引不越界
        x_idx = torch.clamp((all_points[:, 0] + 0.5).long(), 0, GRID_SIZE-1)
        y_idx = torch.clamp((all_points[:, 1] + 0.5).long(), 0, GRID_SIZE-1)
        
        # 计算一维索引
        flat_indices = x_idx * GRID_SIZE + y_idx
        
        # 计算频次（限制最大索引）
        max_index = GRID_SIZE * GRID_SIZE - 1
        flat_indices = torch.clamp(flat_indices, 0, max_index)
        hist = torch.bincount(flat_indices, minlength=GRID_SIZE**2)
        
        # 重塑为二维概率矩阵
        hist = hist[:GRID_SIZE**2].view(GRID_SIZE, GRID_SIZE).float()
        
        # 计算概率分布
        self.env.target_prob = hist / (NUM_TARGET_TRAJ * TARGET_STEPS)
        self.env.target_prob[self.env.grid.bool()] = 0.0

# 遗传算法路径规划（PyTorch实现）
class GeneticPathPlanner(torch.nn.Module):
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.population = None
        self.best_path = None

    def get_valid_neighbors(self, pos):
        # 输入pos形状应为 (2,)
        x, y = pos
        deltas = torch.tensor([[dx, dy] for dx in [-1,0,1] for dy in [-1,0,1] if (dx, dy) != (0,0)])
        neighbors = pos + deltas  # 形状自动广播为 (8, 2)
        
        # 筛选有效位置
        valid_mask = (neighbors >= 0) & (neighbors < GRID_SIZE)
        valid_mask = valid_mask.all(dim=1)
        neighbors = neighbors[valid_mask]
        
        # 进一步筛选非障碍物
        grid_valid = self.env.grid[neighbors[:,0].long(), neighbors[:,1].long()] == 0
        return neighbors[grid_valid]  # 形状为 (N, 2)
        
    def initialize_population(self):
        # 使用张量存储种群（POP_SIZE, PATH_LENGTH, 2）
        population = []
        for _ in range(POP_SIZE):
            path = self.generate_single_path()
            population.append(path)
        self.population = torch.stack(population)
        
   
    def generate_single_path(self):
        # 生成连续且唯一性路径
        path = [self.env.start_pos.clone().float()]
        current_pos = self.env.start_pos.clone()
        visited = set()
        
        for _ in range(PATH_LENGTH-1):
            neighbors = self.get_valid_neighbors(current_pos)
            if len(neighbors) == 0:
                next_pos = current_pos.clone()
            else:
                # 优先选择未访问的邻域
                unvisited = [pos for pos in neighbors if tuple(pos.tolist()) not in visited]
                if unvisited:
                    idx = torch.randint(len(unvisited), (1,)).item()
                    next_pos = unvisited[idx]
                else:
                    idx = torch.randint(len(neighbors), (1,)).item()
                    next_pos = neighbors[idx]
            
            path.append(next_pos)
            visited.add(tuple(next_pos.tolist()))
            current_pos = next_pos.clone()
        
        return torch.stack(path)
    
    def evaluate_fitness(self):
        # 批量计算适应度（GPU加速）
        batch_points = self.population.view(-1, 2).long()
        probs = self.env.target_prob[batch_points[:,0], batch_points[:,1]]
        probs = probs.view(POP_SIZE, PATH_LENGTH)
        
        # 连续性惩罚
        diff = torch.abs(self.population[:,1:] - self.population[:,:-1])
        continuity_penalty = (diff.sum(dim=2) > 1).float().sum(dim=1) * 0.5
        
        # 重复访问惩罚（修正长度问题）
        max_unique = PATH_LENGTH  # 路径固定长度，最大唯一点数为PATH_LENGTH
        unique_counts_list = []
        for p in self.population:
            unique, counts = torch.unique(p, dim=0, return_counts=True)
            padded_counts = torch.nn.functional.pad(counts, (0, max_unique - len(counts)), value=0)  # 填充至固定长度
            unique_counts_list.append(padded_counts)
    
        unique_counts = torch.stack(unique_counts_list)
        repeat_penalty = (unique_counts > 1).sum(dim=1) * 0.2
        
        return probs.sum(dim=1) - continuity_penalty - repeat_penalty
    
    def evolve(self):
        self.initialize_population()
        device = self.env.grid.device
        
        for _ in tqdm(range(GEN_MAX)):
            # 评估适应度
            fitness = self.evaluate_fitness()
            
            # 选择（锦标赛选择）
            selected_idx = torch.zeros(POP_SIZE, dtype=torch.long, device=device)
            for i in range(POP_SIZE):
                candidates = torch.randint(POP_SIZE, (3,))
                selected_idx[i] = candidates[fitness[candidates].argmax()]
            self.population = self.population[selected_idx]
            
            # 交叉（两点交叉）
            crossover_mask = torch.rand(POP_SIZE, device=device) < CROSS_RATE
            crossover_points = torch.randint(1, PATH_LENGTH-1, (2, POP_SIZE))
            for i in range(0, POP_SIZE, 2):
                if crossover_mask[i] and i+1 < POP_SIZE:
                    start, end = crossover_points[:,i].sort().values
                    temp = self.population[i, start:end].clone()
                    self.population[i, start:end] = self.population[i+1, start:end]
                    self.population[i+1, start:end] = temp
            
            # 变异（导向型变异）
            mutate_mask = torch.rand_like(self.population[:,:,0]) < MUTATE_RATE
            for i in range(POP_SIZE):
                for j in range(1, PATH_LENGTH-1):  # 避开起点
                    if mutate_mask[i,j]:
                        neighbors = self.get_valid_neighbors(self.population[i,j-1])
                        if len(neighbors) > 0:
                            # 选择概率最高的邻域
                            neighbors_long = neighbors.long()  # 转换为LongTensor
                            neighbor_probs = self.env.target_prob[neighbors_long[:,0], neighbors_long[:,1]]
                            self.population[i,j] = neighbors[neighbor_probs.argmax()]
            
            # 更新最佳路径
            best_idx = fitness.argmax()
            self.best_path = self.population[best_idx].cpu().numpy()

    # 可视化函数
    def visualize_system(env, search_path):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 障碍物可视化（保持不变）
        ax.set_title("Search Path & Obstacles")
        env_display = torch.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=torch.float32)
        env_display[env.grid.bool()] = torch.tensor([0.2, 0.2, 0.2], dtype=torch.float32)
        ax.imshow(env_display.numpy(), origin='lower')  # 关键修改：设置origin='lower'
        
        # 初始化元素
        path_line, = ax.plot([], [], 'go-', lw=2, markersize=8, markerfacecolor='yellow', markeredgecolor='black')
        current_point = ax.scatter([], [], c='red', s=100, edgecolor='black', zorder=5)
        
        # 增强可视化（保持不变）
        ax.grid(which='both', color='lightgray', linestyle='--', linewidth=0.5)
        ax.set_xticks(np.arange(0, GRID_SIZE, 20))
        ax.set_yticks(np.arange(0, GRID_SIZE, 20))
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")

        def update(frame):
            if frame < len(search_path):
                # 正确提取坐标点（交换行列顺序）
                x_coords = search_path[:frame+1, 1].tolist()  # 列坐标 -> x轴
                y_coords = search_path[:frame+1, 0].tolist()  # 行坐标 -> y轴
                
                # 更新路径线
                path_line.set_data(x_coords, y_coords)
                
                # 更新当前点
                if len(x_coords) > 0:
                    current_point.set_offsets(np.c_[x_coords[-1], y_coords[-1]])
            
            return path_line, current_point

        # 创建动画（帧数需匹配路径长度）
        ani = FuncAnimation(fig, update, 
                        frames=len(search_path),  # 直接使用路径长度
                        interval=100, 
                        blit=True,
                        repeat=False)
        
        ax.legend([path_line, current_point], 
                ['Search Path', 'Current Position'],
                loc='upper right')
        
        plt.tight_layout()
        plt.show()

        # 创建动画
        ani = FuncAnimation(fig, update, 
                        frames=len(search_path)+5,  # +5保持最后帧的显示
                        interval=100, 
                        blit=True,
                        repeat=False)
        
        # 添加图例
        ax.legend([path_line, current_point], 
                ['Search Path', 'Current Position'],
                loc='upper right')
        
        plt.tight_layout()
        plt.show()

def visualize_system(env, target_traj, search_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 环境可视化
    ax1.set_title("Search Environment")
    env_display = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    env_display[env.grid==1] = [0, 0, 0]  # 障碍物黑色
    ax1.imshow(env_display)
    
    # 目标轨迹可视化
    line, = ax1.plot([], [], 'r-', lw=1, alpha=0.3)
    scatter = ax1.scatter([], [], c='r', s=10, alpha=0.5)
    
    # 概率分布可视化
    prob_display = ax2.imshow(env.target_prob.T, cmap='hot', animated=True)
    ax2.set_title("Target Probability Distribution")
    
    # 搜索路径初始化
    search_line, = ax1.plot([], [], 'g-', lw=2)
    search_points = ax1.scatter([], [], c='lime', s=50, edgecolor='black')
    
    def update(frame):
        # 更新目标轨迹
        if frame < TARGET_STEPS:
            x = [traj[frame][1] for traj in target_traj[:100]]
            y = [traj[frame][0] for traj in target_traj[:100]]
            scatter.set_offsets(np.c_[x, y])
        
        # 更新搜索路径
        if frame < len(search_path):
            path_x = [p[1] for p in search_path[:frame+1]]
            path_y = [p[0] for p in search_path[:frame+1]]
            search_line.set_data(path_x, path_y)
            search_points.set_offsets(np.c_[path_x[-1:], path_y[-1:]])
        
        return scatter, search_line, search_points
    
    ani = FuncAnimation(fig, update, frames=max(TARGET_STEPS, PATH_LENGTH),
                        interval=50, blit=True)
    plt.show()

# 主程序
if __name__ == "__main__":
    # 初始化环境（GPU加速）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = SearchEnvironment().to(device)
    
    # 蒙特卡洛目标轨迹生成
    mc = MonteCarloTarget(env).to(device)
    trajectories = mc.generate_trajectories()
    
    # 遗传算法路径规划
    planner = GeneticPathPlanner(env).to(device)
    planner.evolve()
    
    # 可视化
    visualize_system(env.cpu(), trajectories[1:100], planner.best_path)

    