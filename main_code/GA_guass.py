import numpy as np
import matplotlib.pyplot as plt


# 设置全局字体为支持中文的字体（如黑体、微软雅黑等）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'FangSong']  
# 解决负号显示为方块的问题
plt.rcParams['axes.unicode_minus'] = False  

# 参数设置
GRID_SIZE = 20
POP_SIZE = 200
MAX_GEN = 400
MUTATION_RATE = 0.15
GAUSSIAN_CENTER = (8, 8)
GAUSSIAN_SIGMA = 5

np.random.seed(86)

OBSTACLES = [
    (5,5), (2,6), (5,11), (6,5), (6,7), (9,5), (7,6), (7,7),  # 方形障碍物
    (12,1), (3,10), (14,5), (15,7), (12,10),            # 横向障碍
    (8,15), (9,2), (5,15), (13,15)                        # 纵向障碍
]


def initialize_population():
    population = []
    for _ in range(POP_SIZE):
        path = [(2, 2)]
        current = (2, 2)  # 起点修正为 (2,2)
        for _ in range(GRID_SIZE - 1):
            possible_moves = []
            # 检查四个方向是否越界或进入障碍物
            if current[0] < GRID_SIZE-1 and (current[0]+1, current[1]) not in OBSTACLES:
                possible_moves.append('down')
            if current[0] > 0 and (current[0]-1, current[1]) not in OBSTACLES:
                possible_moves.append('up')
            if current[1] < GRID_SIZE-1 and (current[0], current[1]+1) not in OBSTACLES:
                possible_moves.append('right')
            if current[1] > 0 and (current[0], current[1]-1) not in OBSTACLES:
                possible_moves.append('left')
            
            if not possible_moves: 
                break  # 无路可走时终止路径
            move = np.random.choice(possible_moves)

            # 更新坐标
            next_x = current[0] + (1 if move == 'down' else -1 if move == 'up' else 0)
            next_y = current[1] + (1 if move == 'right' else -1 if move == 'left' else 0)
            path.append((next_x, next_y))
            current = (next_x, next_y)
        population.append(path)
    return np.array(population)

def gaussian_prob(x, y):
    return np.exp(-((x - GAUSSIAN_CENTER[0])**2 + (y - GAUSSIAN_CENTER[1])**2)/(2*GAUSSIAN_SIGMA**2))

def is_collision(path):
    for point in path:
        if tuple(point) in OBSTACLES:
            return True
    return False

def fitness(path):
    if is_collision(path):  # 路径碰撞则适应度为0
        return 0
    end_x, end_y = path[-1]
    target_fitness = gaussian_prob(end_x, end_y)
    length_penalty = (len(path)-1) / (GRID_SIZE*2)  
    return max(target_fitness - 0.1 * length_penalty, 0)

def selection(population, fitnesses):
    fitnesses = np.clip(fitnesses, 0, None)  # 去除负值
    if np.sum(fitnesses) == 0:
        fitnesses = np.ones_like(fitnesses)  # 处理全零情况
    probabilities = fitnesses / fitnesses.sum()
    idx = np.random.choice(len(population), size=POP_SIZE//2, p=probabilities)
    return population[idx]

def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1)-1)
    child1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(path):
    if np.random.rand() < MUTATION_RATE and len(path) > 2:
        mutate_pos = np.random.randint(1, len(path)-1)
        # 确保变异后的移动有效
        prev = path[mutate_pos-1]
        next_ = path[mutate_pos+1]
        # 生成新坐标，保持路径连续性
        new_x = prev[0] + (next_[0] - prev[0])
        new_y = prev[1] + (next_[1] - prev[1])
        path[mutate_pos] = (new_x, new_y)
    return path

# 主算法流程
population = initialize_population()
best_path = None

for gen in range(MAX_GEN):
    fitnesses = np.array([fitness(path) for path in population])
    best_idx = np.argmax(fitnesses)
    best_path = population[best_idx]
    
    # 选择
    selected = selection(population, fitnesses)
    
    # 交叉与变异
    new_pop = []
    for i in range(0, len(selected)-1, 2):
        parent1, parent2 = selected[i], selected[i+1]
        child1, child2 = crossover(parent1, parent2)
        new_pop.append(mutate(child1))
        new_pop.append(mutate(child2))
    population = np.array(new_pop)

# 可视化
plt.figure(figsize=(10, 10))
plt.title("高斯分布路径规划")

# 绘制障碍物
obs_x = [x for (x,y) in OBSTACLES]
obs_y = [y for (x,y) in OBSTACLES]
plt.scatter(obs_x, obs_y, c='black', s=200, marker='s', label='障碍物')

# 绘制高斯分布背景
X, Y = np.meshgrid(np.arange(GRID_SIZE), np.arange(GRID_SIZE))
Z = gaussian_prob(X, Y)
plt.contourf(X, Y, Z, levels=20, cmap='Blues', alpha=0.4)

# 绘制最优路径
x_coords = [p[0] for p in best_path]
y_coords = [p[1] for p in best_path]
plt.plot(x_coords, y_coords, 'r-', linewidth=2, marker='o', markersize=4)
plt.scatter(2, 2, c='green', s=200, label='起始点')
plt.scatter(best_path[-1][0], best_path[-1][1], c='blue', s=200, marker='*', label='终止点')

plt.xticks(np.arange(0, GRID_SIZE, 1))
plt.yticks(np.arange(0, GRID_SIZE, 1))
plt.grid(True)
plt.legend()
plt.show()