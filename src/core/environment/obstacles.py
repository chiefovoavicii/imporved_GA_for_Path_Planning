import numpy as np
from src.core.environment.grid import GridEnvironment

class ObstacleManager:
    def __init__(self, config):
        """
        障碍物管理类

        参数:
            obstacles: 障碍物地图
            obstacle_type: 障碍物类型 ('fixed' 或 'random')
            obstacle_density: 随机障碍物密度 (仅当obstacle_type='random'时有效)
        """
        #self.obstacles = np.zeros((config["grid_size"], config["grid_size"]), dtype=int)  # 障碍物地图矩阵
        self.obstacles = np.zeros((config["grid_length"], config["grid_width"]), dtype=int)
        self.obstacle_type = config["obstacle_type"]
        self.obstacle_density = float(config["obstacle_density"])
        self.grid = GridEnvironment(config)


    def initialize_obstacles(self):
        """初始化障碍物"""
        if self.obstacle_type == 'fixed':
            self.initialize_fixed_obstacles()
        elif self.obstacle_type == 'random':
            self.initialize_random_obstacles()

    def initialize_fixed_obstacles(self):
        """固定障碍物配置(length*width网格)"""
        fixed_obstacles = [
            (0, 4), (0, 10), (0, 12), (1, 0), (1, 6), (1, 7), (1, 8), (2, 13), (2, 20),
            (5, 1), (5, 5), (5, 6), (5, 7), (5, 18), (7, 7), (8, 17), (8, 18), (8, 19),
            (9, 5), (9, 14), (9, 15), (10, 0), (11, 9), (11, 12), (12, 3), (12, 13),
            (13, 6), (13, 8), (13, 19), (13, 20), (13, 21), (14, 21), (14, 22)
            ]
        """
        fixed_obstacles = [
            (0, 4), (0, 10), (0, 12), (1, 0), (1, 6), (1, 7), (1, 8), (2, 13),
            (5, 1), (5, 5), (5, 6), (5, 7), (7, 7), (8, 15), (9, 5), (9, 14),
            (10, 0), (11, 9), (11, 12), (12, 13), (13, 6), (13, 8), (13, 19),
            (14, 6), (15, 2), (15, 7), (15, 15), (15, 18), (16, 1), (16, 4), (16, 16), (16, 17),
            (17, 8), (17, 14), (17, 15), (18, 2), (19, 1)
        ]
        """
        for x, y in fixed_obstacles:
            if 0 <= x < self.grid.grid_length and 0 <= y < self.grid.grid_width:
                self.obstacles[x, y] = 1

    def initialize_random_obstacles(self):
        """随机障碍物生成（适用于任何n*n网格）"""
        num_obstacles = int(self.obstacle_density * self.grid.grid_length * self.grid.grid_width)
        indices = np.random.choice(self.grid.grid_length * self.grid.grid_width, num_obstacles, replace=False)
        for idx in indices:
            x, y = np.unravel_index(idx, (self.grid.grid_length, self.grid.grid_width))
            self.obstacles[x, y] = 1

    def ensure_start_pos_clear(self, start_pos):
        """确保起点无障碍物"""
        number = len(start_pos)  # 计算起点数量
        for i in range(number):
            x, y = start_pos[i]
            if 0 <= x < self.grid.grid_length and 0 <= y < self.grid.grid_width:
                self.obstacles[x, y] = 0

    def get_obstacle_grid(self):
        """获取障碍物网格"""
        return self.obstacles.copy()