import numpy as np

class GridEnvironment:
	"""网格环境基础类"""

	def __init__(self, config):

		self.grid_size = config["grid_size"]  # 地图大小
		self.grid_length = config["grid_length"]  # 地图长度
		self.grid_width = config["grid_width"]  # 地图宽度
		self.dirs = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]  # 8个方向动作

		self.current_time = 0.0  # 当前训练时间
		self.total_time = config["total_time"]  # 总时长
		self.max_steps = config["max_steps"]  # 最大步数
		self.time_step = self.total_time / self.max_steps

	def update_time(self):
		"""更新时间"""
		self.current_time += self.time_step
		if self.current_time > self.total_time:
			self.current_time = self.total_time



