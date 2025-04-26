'''
生成环境网格和障碍物
'''


class GridEnvironment:
	"""网格环境基础类"""

	def __init__(self, config):

		self.grid_size = config["grid_size"] # 地图大小
		self.dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # 8个方向动作

		self.start_pos = config["start_pos"]


