#均匀分布路径规划
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为支持中文的字体（如黑体、微软雅黑等）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False 

# 参数设置（保留原始参数）
GRID_SIZE = 20
POP_SIZE = 100
GEN_MAX = 200
PATH_LENGTH = 30
OBSTACLE_NUM = 8
START_POS = np.array([2, 2])
MUTATION_RATE = 0.2

np.random.seed(26)

class SearchEnvironment:
    def __init__(self):
        # 生成随机障碍物
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        for _ in range(OBSTACLE_NUM):
            x, y = np.random.randint(1, GRID_SIZE-1, 2)
            self.grid[max(0,x-1):x+1, max(0,y-1):y+1] = 1
            
        # 生成均匀目标概率图
        self.target_prob = np.ones_like(self.grid, dtype=np.float32)
        self.target_prob[self.grid == 1] = 0.0  # 障碍物区域清零
        self.target_prob /= self.target_prob.sum()  # 归一化

class GeneticPathPlanner:
    def __init__(self, env):
        self.env = env
        self.population = []
        self.best_path = None

    def get_neighbors(self, pos):
        """8邻域坐标生成（保留障碍物检测）"""
        deltas = np.array([[-1,0], [1,0], [0,-1], [0,1],
                         [-1,-1], [-1,1], [1,-1], [1,1]])
        neighbors = pos + deltas
        valid = (neighbors >= 0) & (neighbors < GRID_SIZE)
        valid = valid.all(axis=1)
        valid_neighbors = neighbors[valid]
        obstacle_free = self.env.grid[valid_neighbors[:,0], valid_neighbors[:,1]] == 0
        return valid_neighbors[obstacle_free]

    def generate_path(self):
        """改进路径生成策略"""
        path = [START_POS.copy()]
        current = START_POS.copy()
        
        for _ in range(PATH_LENGTH-1):
            neighbors = self.get_neighbors(current)
            if len(neighbors) == 0:
                current = np.clip(current + np.random.randint(-1,2,2), 0, GRID_SIZE-1)
            else:
                # 均匀概率选择
                probs = self.env.target_prob[neighbors[:,0], neighbors[:,1]]
                probs /= probs.sum()
                current = neighbors[np.random.choice(len(neighbors), p=probs)]
            
            path.append(current.copy())
                
        while len(path) < PATH_LENGTH:
            path.append(path[-1].copy())
        return np.array(path)

    def evaluate(self, path):
        """适应度函数改造"""
        # 路径有效性检查
        valid = np.all(self.env.grid[path[:,0].astype(int), path[:,1].astype(int)] == 0)
        
        # 新增覆盖度指标（替代目标接近度）
        unique_coverage = len(np.unique(path, axis=0)) / (GRID_SIZE**2 - OBSTACLE_NUM)
        
        # 路径平滑度
        dir_changes = np.linalg.norm(path[1:] - path[:-1], axis=1)
        smoothness = 1 / (1 + np.std(dir_changes))
        
        # 路径重复率惩罚
        _, counts = np.unique(path, axis=0, return_counts=True)
        repeat_penalty = np.mean(counts > 1)
        
        return (valid * 1000 + 
                unique_coverage * 800 + 
                smoothness * 300 - 
                repeat_penalty * 500)

    def evolve(self):
        """进化流程优化"""
        # 初始化种群
        self.population = [self.generate_path() for _ in range(POP_SIZE)]
        
        for gen in range(GEN_MAX):
            # 评估适应度
            fitness = np.array([self.evaluate(p) for p in self.population])
            
            # 锦标赛选择
            selected = []
            for _ in range(POP_SIZE):
                candidates = np.random.choice(POP_SIZE, 3)
                winner = candidates[np.argmax(fitness[candidates])]
                selected.append(self.population[winner])
            
            # 两点交叉与变异
            new_pop = []
            for i in range(0, POP_SIZE, 2):
                p1, p2 = selected[i], selected[i+1]
                pt1, pt2 = sorted(np.random.randint(1, PATH_LENGTH-1, 2))
                child1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
                child2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
                new_pop.extend([child1, child2])

            # 导向型变异
            for i in range(POP_SIZE):
                if np.random.rand() < MUTATION_RATE:
                    mut_point = np.random.randint(1, PATH_LENGTH-1)
                    neighbors = self.get_neighbors(new_pop[i][mut_point-1])
                    if len(neighbors) > 0:
                        new_pop[i][mut_point] = neighbors[np.argmax(
                            self.env.target_prob[neighbors[:,0], neighbors[:,1]])]
            
            # 精英保留策略
            self.population = new_pop[:POP_SIZE-1] + [self.population[np.argmax(fitness)]]

        self.best_path = self.population[np.argmax(fitness)]
        return self.best_path

# 可视化部分（保留原始实现）
env = SearchEnvironment()
planner = GeneticPathPlanner(env)
best_path = planner.evolve()

plt.figure(figsize=(10,10))
plt.imshow(env.grid, cmap='Greys', origin='lower', 
          extent=[0, GRID_SIZE, 0, GRID_SIZE])

# 绘制路径
unique_path = np.unique(best_path, axis=0)
plt.plot(unique_path[:,1]+0.5, unique_path[:,0]+0.5, 'b-', lw=2)
plt.scatter(START_POS[1]+0.5, START_POS[0]+0.5, 
           c='lime', s=150, edgecolors='darkgreen')

plt.title('均匀分布路径规划', fontsize=14)
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.show()