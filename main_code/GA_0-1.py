#0-1分布
import numpy as np
import matplotlib.pyplot as plt

# 设置全局字体为支持中文的字体（如黑体、微软雅黑等）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  

# 参数设置
GRID_SIZE = 20
POP_SIZE = 100
GEN_MAX = 200
PATH_LENGTH = 25
OBSTACLE_NUM = 8
START_POS = np.array([2, 2])
TARGET_NUM = 1
MUTATION_RATE = 0.2  # 提升变异概率

np.random.seed(66)

class SearchEnvironment:
    def __init__(self):
        # 生成随机障碍物（移除固定随机种子）
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        for _ in range(OBSTACLE_NUM):
            x, y = np.random.randint(0, GRID_SIZE-5, 2)
            self.grid[max(0,x-1):x+1, max(0,y-1):y+1] = 1
            
        # 生成随机目标点
        self.targets = []
        while len(self.targets) < TARGET_NUM:
            pos = np.random.randint(0, GRID_SIZE, 2)
            if self.grid[pos[0], pos[1]] == 0:
                self.targets.append(pos)
        
        # 目标概率图
        self.target_prob = np.zeros_like(self.grid, dtype=np.float32)
        for t in self.targets:
            self.target_prob[t[0], t[1]] = 1.0

class GeneticPathPlanner:
    def __init__(self, env):
        self.env = env
        self.population = []
        self.best_path = None

    def get_neighbors(self, pos):
        """获取8邻域有效坐标（增加随机性）"""
        deltas = np.array([[-1,0], [1,0], [0,-1], [0,1],
                         [-1,-1], [-1,1], [1,-1], [1,1]])
        neighbors = pos + deltas
        valid = (neighbors >= 0) & (neighbors < GRID_SIZE)
        valid = valid.all(axis=1)
        valid_neighbors = neighbors[valid]
        obstacle_free = self.env.grid[valid_neighbors[:,0], valid_neighbors[:,1]] == 0
        return valid_neighbors[obstacle_free]

    def generate_path(self):
        """改进路径生成（增加随机探索）"""
        path = [START_POS.copy()]
        current = START_POS.copy()
        
        for _ in range(PATH_LENGTH-1):
            neighbors = self.get_neighbors(current)
            if len(neighbors) == 0:
                # 随机方向修复
                current = np.clip(current + np.random.randint(-1,2,2), 0, GRID_SIZE-1)
            else:
                # 引入概率选择（参考网页2的轮盘赌思想）
                probs = self.env.target_prob[neighbors[:,0], neighbors[:,1]] + 0.1  # 防止零概率
                probs /= probs.sum()
                current = neighbors[np.random.choice(len(neighbors), p=probs)]
            
            path.append(current.copy())
            if any(np.all(current == t) for t in self.env.targets):
                break
                
        while len(path) < PATH_LENGTH:
            path.append(path[-1].copy())
        return np.array(path)

    def evaluate(self, path):
        """改进适应度函数（参考网页2的惩罚机制）"""
        # 路径有效性检查（严格碰撞检测）
        valid = np.all(self.env.grid[path[:,0].astype(int), path[:,1].astype(int)] == 0)
        
        # 目标接近度（多目标处理）
        min_dists = [np.min(np.linalg.norm(path - t, axis=1)) for t in self.env.targets]
        proximity = np.exp(-np.mean(min_dists))
        
        # 路径长度惩罚
        unique_path = np.unique(path, axis=0, return_index=True)[0]
        length_penalty = len(unique_path)/PATH_LENGTH
        
        # 方向变化惩罚（参考网页44）
        dir_changes = np.linalg.norm(path[1:] - path[:-1], axis=1)
        smoothness = 1 / (1 + np.std(dir_changes))
        
        return (valid * 1000 + 
                proximity * 500 + 
                smoothness * 200 + 
                length_penalty * 300)

    def evolve(self):
        """改进进化流程（参考网页34的混合算法思想）"""
        # 初始化多样化种群
        self.population = [self.generate_path() for _ in range(POP_SIZE)]
        best_fitness = -float('inf')
        
        for gen in range(GEN_MAX):
            # 评估适应度
            fitness = np.array([self.evaluate(p) for p in self.population])
            
            # 动态记录最优（参考网页44）
            current_best_idx = np.argmax(fitness)
            if fitness[current_best_idx] > best_fitness:
                best_fitness = fitness[current_best_idx]
                self.best_path = self.population[current_best_idx].copy()
                #print(f"Gen {gen}: New best fitness {best_fitness:.1f}")
            
            # 锦标赛选择（参考网页2）
            selected = []
            for _ in range(POP_SIZE):
                candidates = np.random.choice(POP_SIZE, 3)
                winner = candidates[np.argmax(fitness[candidates])]
                selected.append(self.population[winner])
            
            # 顺序交叉（参考网页63的路径连续性要求）
            new_pop = []
            for i in range(0, POP_SIZE, 2):
                p1, p2 = selected[i], selected[i+1]
                if np.random.rand() < 0.8:  # 交叉概率
                    # 两点交叉
                    pt1, pt2 = sorted(np.random.randint(1, PATH_LENGTH-1, 2))
                    child1 = np.concatenate([p1[:pt1], p2[pt1:pt2], p1[pt2:]])
                    child2 = np.concatenate([p2[:pt1], p1[pt1:pt2], p2[pt2:]])
                    new_pop.extend([child1, child2])
                else:
                    new_pop.extend([p1, p2])
            
            # 增强变异（参考网页64的路径优化）
            for i in range(POP_SIZE):
                if np.random.rand() < MUTATION_RATE:
                    # 多类型变异
                    mut_type = np.random.choice(['replace', 'insert', 'delete'])
                    # 替换变异
                    if mut_type == 'replace' and len(new_pop[i]) > 2:
                        mut_point = np.random.randint(1, len(new_pop[i])-1)
                        neighbors = self.get_neighbors(new_pop[i][mut_point-1])
                        if len(neighbors) > 0:
                            new_pop[i][mut_point] = neighbors[np.random.randint(len(neighbors))]
                    # 插入变异
                    elif mut_type == 'insert' and len(new_pop[i]) < PATH_LENGTH:
                        mut_point = np.random.randint(1, len(new_pop[i]))
                        new_point = (new_pop[i][mut_point-1] + new_pop[i][mut_point])//2
                        new_pop[i] = np.insert(new_pop[i], mut_point, new_point, axis=0)
                    # 删除变异    
                    elif mut_type == 'delete' and len(new_pop[i]) > 2:
                        del_point = np.random.randint(1, len(new_pop[i])-1)
                        new_pop[i] = np.delete(new_pop[i], del_point, axis=0)
            
            # 精英保留+随机注入（参考网页34）
            self.population = new_pop[:POP_SIZE-2] + [self.best_path, self.generate_path()]
            
        return self.best_path

# 运行与可视化
env = SearchEnvironment()
planner = GeneticPathPlanner(env)
best_path = planner.evolve()

plt.figure(figsize=(10,10))
plt.imshow(env.grid, cmap='Greys', origin='lower', 
          extent=[0, GRID_SIZE, 0, GRID_SIZE])

# 绘制目标点
for t in env.targets:
    plt.scatter(t[1]+0.5, t[0]+0.5, c='gold', s=200, marker='*', edgecolors='red')

# 绘制路径
unique_path = np.unique(best_path, axis=0)
plt.plot(unique_path[:,1]+0.5, unique_path[:,0]+0.5, 'b-', lw=2)
plt.scatter(START_POS[1]+0.5, START_POS[0]+0.5, 
           c='lime', s=150, edgecolors='darkgreen')

plt.title('0-1分布路径规划', fontsize=14)
plt.xlabel('X坐标')
plt.ylabel('Y坐标')
plt.show()