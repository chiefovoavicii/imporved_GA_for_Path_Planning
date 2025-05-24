# ui/config_parser.py
import json
import os
from copy import deepcopy

# 默认配置路径
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "default_config.json")

config = {
	# ===========================网格设置===========================
	'grid_size': 100,  # 网格大小
	'grid_length': 15,  # 网格长度
	'grid_width': 23,  # 网格宽度
	'obstacle_type': 'fixed',  # 障碍物分布形式，可选'fixed'固定或'random'随机
	'obstacle_density': 0.01,  # 随机障碍物密度

	# ===========================搜索平台设置===========================
	'num_plane': [(1, 4), (2, 16)],  # 飞机数量
	'num_ship': [(1, 16), (9, 3)],  # 船舶数量

	# ===========================目标存在概率分布===========================
	'pod': 0.8,  # POD值(平台探测概率)，POD∈[0,1]
	'poc_type': 'dynamic_gaussian',  # 目标存在概率，可选'gaussian'或'01'或'uniform'或'dynamic_gaussian'
	'target_pos_01': (1, 19),  # 01分布中“1”分布的位置（目标位置）
	'gauss_center': (13, 5),  # 高斯分布中心'(x,y)'
	'theta': 0.3,  # 高斯分布低价值区域阈值，不使用高斯分布时请设为0，用于强化学习算法
	'sigma': 5.0,  # 高斯分布标准差

	# ===========================算法选择===========================
	'algorithm': 'rl',  # 搜索算法，可选'rl'或'greedy'或'genetic'

	# ===========================训练设置===========================
	'max_steps': 15,  # 每回合最大步数
	'episodes': 100,  # 训练回合数
	'total_time': 7200,  # 总训练时间
}

def load_config(config_path=None):
    """加载配置（优先从文件，无文件时用默认值）"""
    # 基础默认配置
    default_config = {
        'grid_size': 20,
        'pod': 0.8,
        'poc_type': 'gaussian',
        'obstacle_type': 'fixed',
        'obstacle_density': 0.1,
        'start_pos': (2, 8),
        'target_pos_01': (1, 19),
        'gauss_center': (16, 6),
        'theta': 0.3,
        'sigma': 5.0,
        'algorithm': 'genetic',
        'max_steps': 1000,
        'episodes': 1000,
    }