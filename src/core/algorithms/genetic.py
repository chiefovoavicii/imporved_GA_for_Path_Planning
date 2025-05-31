import numpy as np
import random
import datetime
from src.ui.config_parser import config
import time 

class GeneticAgent:
    def __init__(self, env, gp, obs, 
                 population_size=100, 
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 elite_ratio=0.25):
        """
        遗传算法智能体
        :param population_size: 种群规模 
        :param crossover_rate: 交叉概率 
        :param mutation_rate: 变异概率 
        :param elite_ratio: 精英保留比例 
        """
          # 现在是一个位置列表
        self.num_agents = len(config["num_plane"]) + len(config["num_ship"])  # 智能体数量

        self.env = env
        self.grid_state = np.where(obs.obstacles == 1, -2, 0).astype(int)
        self.population = []
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_num = int(population_size * elite_ratio)
        self.gp = gp
        self.start_pos_nums = config["num_plane"] + config["num_ship"]
        # 环境参数
        
        self.max_steps = config["max_steps"]
        
        self.grid_x = config["grid_x"]
        self.grid_y = config["grid_y"]

        # 路径编码参数
        self.chromosome_length = self.max_steps  # 染色体长度=最大步数 
        self.gene_pool = self._create_gene_pool()  # 可行基因库
        
        # 初始化种群 
        self._initialize_population()

    def _generate_single_path(self, start_pos):
        """为单个智能体生成路径"""
        path = [start_pos]
        visited = {start_pos}
        current_pos = start_pos
        
        for _ in range(self.chromosome_length-1):
            candidates = [pos for pos in self.gene_pool.get(current_pos, []) 
                        if pos not in visited]
            if not candidates:
                break
            # 修改点2：优先朝向未探索区域移动
            next_pos = max(candidates, 
                        key=lambda p: len([n for n in self.gene_pool[p] 
                                         if n not in visited]))
            path.append(next_pos)
            visited.add(next_pos)
            current_pos = next_pos
        return path

    def _generate_individual(self):
        """生成包含多个路径的个体"""
        individual = []
        shared_visited = set()
        
        for start in self.start_pos_nums:
            path = [start]
            current = start
            for step in range(self.chromosome_length-1):
                candidates = self.gene_pool.get(current, [])
                if not candidates:
                    break
                    
                # 动态选择策略：前期探索，后期趋向高概率区域
                if step < 0.5 * self.chromosome_length:
                    next_pos = max(candidates, key=lambda p: len(self.gene_pool[p]))
                else:
                    next_pos = max(candidates, key=lambda p: self.gp.poc[p[0]][p[1]])
                    
                path.append(next_pos)
                current = next_pos
            individual.append(path)
        return individual



   
    def _create_gene_pool(self):
        """创建可行移动基因库"""
        pool = {}
        for (x, y) in np.argwhere(self.grid_state != -2).astype(int):  # 仅遍历可行点
            valid_moves = []
            # 优先非回头方向（参考网页7的路径方向性启发）
            for dx, dy in [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]:  # 上下右左顺序
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.grid_x and 0 <= ny < self.grid_y:
                    if self.grid_state[nx, ny] != -2:
                        valid_moves.append((int(nx), int(ny)))
            pool[(int(x), int(y))] = valid_moves
        return pool
    '''
    def _generate_individual(self):
        """生成可行个体 """
        path = [self.start_pos]
        visited = {self.start_pos}  # 记录已访问节点
        current_pos = self.start_pos
        
        for _ in range(self.chromosome_length-1):
            # 获取未访问的可行移动方向（参考网页10的启发式搜索）
            candidates = [pos for pos in self.gene_pool.get(current_pos, []) 
                        if pos not in visited]
            
            if not candidates:
                break  # 提前终止路径
                
            # 选择最优移动（结合网页7的启发式评估）
            next_pos = max(candidates, 
                        key=lambda p: len(self.gene_pool[p]) - (0.5 if p in visited else 0))
            
            path.append(next_pos)
            visited.add(next_pos)
            current_pos = next_pos
                
        return path
    '''
    def _initialize_population(self): 
        fail_count = 0
        while len(self.population) < self.pop_size and fail_count < 10000:
            ind = self._generate_individual()
            if self._is_valid_path(ind):
                self.population.append(ind)
            else:
                fail_count += 1
        if len(self.population) < self.pop_size:
            print("警告：无法生成足够初始种群，请检查障碍物配置")

    def _is_valid_path(self, path):
        """路径有效性验证 """
        for pos in path:
            # 显式解包两层元组结构 (常见于路径算法坐标嵌套)
            real_pos = pos[0] if isinstance(pos[0], (tuple, list)) else pos
            x, y = int(real_pos[0]), int(real_pos[1])
            if self.grid_state[x, y] == -2:
                return False
        return len(path) >= 2

    def fitness(self, individual):
        """多路径适应度计算"""
        all_positions = []
        valid = True
        target_score = sum(
        self.gp.poc[x][y] * (1 - step/len(path))  # 越靠近路径末端，权重越高
        for path in individual 
        for step, (x, y) in enumerate(path)
            )
        # 验证所有路径的有效性
        for path in individual:
            if not self._is_valid_path(path):
                valid = False
            all_positions.extend(path)
        if not valid:
            return 0
        
        # 计算协同覆盖率
        unique_coverage = len(set(all_positions))
        coverage = unique_coverage / (self.grid_x*self.grid_y)
        
        # 计算路径效率
        total_length = sum(len(path) for path in individual)
        avg_length = total_length / self.num_agents
        
        # 重复惩罚（智能体间的重叠）
        repeats = len(all_positions) - unique_coverage
        
        # 协同奖励（覆盖区域分布均匀性）
        grid_counts = np.zeros((self.grid_x, self.grid_y))
        for path in individual:
            for (x, y) in path:
                grid_counts[x][y] += 1
        uniformity = 1 / (np.std(grid_counts[grid_counts > 0]) + 1e-6)
        
        # 概率增益
        #gp_reward = sum(self.gp.poc[x][y] for (x, y) in set(all_positions))
        
        # 综合适应度公式
        return (
        target_score * 1.0 +          # 目标导向
        coverage * 0.3 +              # 覆盖率
        - avg_length * 0.01 +         # 路径长度惩罚
        - repeats * 0.5 +             # 重复区域惩罚
        uniformity * 0.1              # 分布均匀性
        )

    def _selection(self):
        """锦标赛选择 """
        selected = []
        for _ in range(self.pop_size - self.elite_num):
            candidates = random.sample(self.population, 3)
            winner = max(candidates, key=lambda x: self.fitness(x))
            selected.append(winner)
        return selected

    def _crossover(self, parent1, parent2):
        """多点交叉策略"""
        child1, child2 = [], []
        
        # 对每个智能体的路径进行交叉
        for p1_path, p2_path in zip(parent1, parent2):
            if random.random() < self.crossover_rate:
                # 寻找最佳交叉点
                common = sorted(
                    [pos for pos in p1_path if pos in set(p2_path)],
                    key=lambda p: -self.gp.poc[p[0]][p[1]]  # 按概率降序排序
                )
                if len(common) >= 2:
                # 排除首尾的中间点策略
                    cross_points = common[1:-1] if len(common) > 2 else common
                
                    try:
                        cross_point = random.choice(cross_points)
                        idx1 = p1_path.index(cross_point)
                        idx2 = p2_path.index(cross_point)
                        
                        # 执行交叉操作
                        child1.append(p1_path[:idx1] + p2_path[idx2:])
                        child2.append(p2_path[:idx2] + p1_path[idx1:])
                    except IndexError:
                        child1.append(p1_path)
                        child2.append(p2_path)
                else:
                    child1.append(p1_path)
                    child2.append(p2_path)
            else:
                child1.append(p1_path)
                child2.append(p2_path)
    
        return child1, child2

    def _mutation(self, individual, episode):
        """协同变异策略"""
        mutated = []
        shared_visited = set()
        current_mutation_rate = self.mutation_rate * (1 - episode/config["episodes"])
        for path in individual:
            # 第一次遍历收集共享信息
            shared_visited.update(path)
        
        for path in individual:
            new_path = path.copy()
            # 变异概率分段应用
            for i in range(1, len(path)-1):
                if random.random() < current_mutation_rate * 0.5:
                    current = new_path[i-1]
                    # 优先选择未被其他智能体访问的位置
                    candidates = sorted(
                    [p for p in self.gene_pool.get(current, [])],
                    key=lambda p: -self.gp.poc[p[0]][p[1]]
                )[:3]  # 选择概率最高的3个邻域点
                    
                    if candidates:
                        new_pos = random.choice(candidates)
                        # 保持路径连续性
                        if new_pos in self.gene_pool.get(current, []):
                            new_path[i] = new_pos
                            shared_visited.add(new_pos)
            mutated.append(new_path)
        return mutated

    @staticmethod
    def _remove_duplicates(path):
        """路径去重工具"""
        seen = set()
        return [x for x in path if not (x in seen or seen.add(x))]

    def evolve(self, episode):
        """进化一代 """
        # 精英保留
        elites = sorted(self.population, key=lambda x: self.fitness(x), reverse=True)[:self.elite_num]
        
        # 选择
        selected = self._selection()
        
        # 交叉
        children = []
        for i in range(0, len(selected)-1, 2):
            child1, child2 = self._crossover(selected[i], selected[i+1])
            children.extend([child1, child2])
        
        # 变异
        mutated = [self._mutation(ind, episode) for ind in children]
        
        # 生成新一代
        self.population = elites + mutated[:self.pop_size - self.elite_num]
        
    def get_best_path(self):
        """获取当前最优路径 """
        return max(self.population, key=lambda x: self.fitness(x))


def train_ga(config, env, gp, obs):
    """遗传算法训练函数"""
    agent = GeneticAgent(env, gp, obs)
    
    print(f"遗传算法训练开始")
    start_time = time.time()
    
    for episode in range(config["episodes"]):
        agent.evolve(episode)
        
        if episode % 100 == 0:
            best = agent.get_best_path()
            unique_coords = {tuple(pos) for path in best for pos in path}
            coverage = len(unique_coords) / (env.grid_x*env.grid_y)
            print(f"Gen {episode} | Coverage: {coverage:.2%} | Length: {[len(x) for x in best]}")
    
    print(f"训练耗时: {time.time() - start_time:.1f}秒")
    return agent

def generate_ga_path(agent):
    """生成多智能体路径"""
    best_paths = agent.get_best_path()
    
    # 优化路径并保持同步
    optimized = []
    shared_visited = set()
    
    # 先收集所有关键点
    all_points = []
    for path in best_paths:
        all_points.extend(path)
    unique_points = [p for p in all_points if p not in shared_visited]
    
    # 重新分配路径
    for i in range(len(best_paths)):
        optimized_path = []
        current = agent.start_pos_nums[i]
        optimized_path.append(current)
        shared_visited.add(current)
        
        for _ in range(agent.max_steps-1):
            # 优先选择未访问的关键点
            candidates = [p for p in agent.gene_pool.get(current, [])
                        if p in unique_points and p not in shared_visited]
            
            if not candidates:
                candidates = agent.gene_pool.get(current, [])
            
            next_pos = max(candidates, 
                         key=lambda p: len(agent.gene_pool[p]) - 
                         (10 if p in shared_visited else 0),
                         default=None)
            
            if next_pos is None:
                break
                
            optimized_path.append(next_pos)
            shared_visited.add(next_pos)
            current = next_pos
        
        optimized.append(optimized_path[:agent.max_steps])
    
    return optimized
