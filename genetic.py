import numpy as np
import random
import datetime
from src.ui.config_parser import config
import time 

class GeneticAgent:
    def __init__(self, env, gp, obs, 
                 population_size=200, 
                 crossover_rate=0.8,
                 mutation_rate=0.1,
                 elite_ratio=0.2):
        """
        遗传算法智能体
        :param population_size: 种群规模 
        :param crossover_rate: 交叉概率 
        :param mutation_rate: 变异概率 
        :param elite_ratio: 精英保留比例 
        """
        self.env = env
        self.grid_state = np.where(obs.obstacles == 1, -2, 0).astype(int)
        self.population = []
        self.pop_size = population_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elite_num = int(population_size * elite_ratio)
        self.gp = gp
        
        # 环境参数
        self.start_pos = config["start_pos"]
        self.max_steps = config["max_steps"]
        self.grid_size = config["grid_size"]
        
        # 路径编码参数
        self.chromosome_length = self.max_steps  # 染色体长度=最大步数 
        self.gene_pool = self._create_gene_pool()  # 可行基因库
        
        # 初始化种群 
        self._initialize_population()

    def _create_gene_pool(self):
        """创建可行移动基因库"""
        pool = {}
        for (x, y) in np.argwhere(self.grid_state != -2):  # 仅遍历可行点
            valid_moves = []
            # 优先非回头方向（参考网页7的路径方向性启发）
            for dx, dy in [(-1,0), (1,0), (0,1), (0,-1)]:  # 上下右左顺序
                nx, ny = x+dx, y+dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.grid_state[nx, ny] != -2:
                        valid_moves.append((nx, ny))
            pool[(x, y)] = valid_moves
        return pool

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
            x, y = pos
            if self.grid_state[x, y] == -2:
                return False
        return len(path) >= 2

    def fitness(self, individual):
        """适应度函数设计 """
        
        repeat_num = len(individual) - len(set(individual))
        # 1.路径有效性惩罚
        if not self._is_valid_path(individual):
            return 0
        
        # 2.覆盖率计算
        coverage = len(set(individual)) / (self.grid_size**2)
        
        # 3.路径长度奖励
        path_length = len(individual)
        length_score =  - (path_length / self.chromosome_length)
        


        gp_reward = 0
        for pos in set(individual):
            x, y = pos
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                gp_reward += self.gp.poc[x][y]  # 累加路径点的概率值

        # 4.障碍物规避
        obstacle_penalty = sum(1 for pos in individual if self.grid_state[pos] == -2)
        

        # 综合适应度 
        return (coverage * 0.2 + 
                length_score * 3 +
                gp_reward * 1.2 -
                repeat_num * 0.5 - 
                obstacle_penalty * 0.8)

    def _selection(self):
        """锦标赛选择 """
        selected = []
        for _ in range(self.pop_size - self.elite_num):
            candidates = random.sample(self.population, 3)
            winner = max(candidates, key=lambda x: self.fitness(x))
            selected.append(winner)
        return selected

    def _crossover(self, parent1, parent2):
        """顺序交叉 """
         # 仅保留公共节点交叉（参考网页4的合法个体修复）
        common_nodes = list(set(parent1) & set(parent2))
        if len(common_nodes) < 2:
            return parent1.copy(), parent2.copy()
        
        # 选择交叉点（避免重复区域）
        crossover_point = random.choice(common_nodes[1:-1])
        idx1 = parent1.index(crossover_point)
        idx2 = parent2.index(crossover_point)
        
        # 重组路径段（参考网页12的基因重组策略）
        child1 = self._remove_duplicates(parent1[:idx1] + parent2[idx2:])
        child2 = self._remove_duplicates(parent2[:idx2] + parent1[idx1:])
        return child1, child2

    def _mutation(self, individual):
         # 插入未访问节点（参考网页9的路径优化策略）
        if len(individual) < 3:
            return individual
        
        # 选择变异位置
        mut_pos = random.randint(1, len(individual)-1)
        current = individual[mut_pos-1]
        
        # 获取可行新节点
        candidates = [pos for pos in self.gene_pool.get(current, [])
                    if pos not in individual]
        
        if candidates:
            # 插入最佳候选（参考网页7的启发式插入）
            best_candidate = max(candidates, 
                            key=lambda p: len(self.gene_pool[p]))
            return individual[:mut_pos] + [best_candidate] + individual[mut_pos:]
        return individual

    @staticmethod
    def _remove_duplicates(path):
        """路径去重工具"""
        seen = set()
        return [x for x in path if not (x in seen or seen.add(x))]

    def evolve(self):
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
        mutated = [self._mutation(ind) for ind in children]
        
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
        agent.evolve()
        
        if episode % 100 == 0:
            best = agent.get_best_path()
            coverage = len(set(best)) / (env.grid_size**2)
            print(f"Gen {episode} | Coverage: {coverage:.2%} | Length: {len(best)}")
    
    print(f"训练耗时: {time.time() - start_time:.1f}秒")
    return agent

def generate_ga_path(agent):
    """生成最优路径"""
    best_path = agent.get_best_path()
    
    # 路径优化：去除重复点 
    optimized = []
    visited = set()
    for pos in best_path:
        if pos not in visited:
            optimized.append(pos)
            visited.add(pos)
    return optimized[:agent.max_steps]  # 截断到最大步长
