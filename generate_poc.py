'''
生成目标存在概率分布:正态分布、均匀分布、0-1分布
'''

import numpy as np
from src.ui.config_parser import config
from src.core.environment.poc.gaussian import initialize_gauss_poc
from src.core.environment.poc.uniform import initialize_uniform_poc
from src.core.environment.poc.zero_one import initialize_01_poc




class GeneratePOC:
	def __init__(self, env):
		# 绑定需要的配置参数到实例属性
		self.env = env

		self.poc_type = config["poc_type"]
		self.poc = np.zeros((config["grid_size"], config["grid_size"])) # 初始化poc矩阵
		self.pod = config["pod"] # 初始化pod
		self.pos = np.zeros((config["grid_size"], config["grid_size"])) # 初始化pos矩阵

		self.target_pos_01 = config["target_pos_01"]
		self.gauss_center = config["gauss_center"]
		self.sigma = config["sigma"]
		self.theta = config["theta"]

		self.initialize_gauss_poc = initialize_gauss_poc
		self.initialize_uniform_poc = initialize_uniform_poc
		self.initialize_01_poc = initialize_01_poc

	def initialize_poc(self):
		"""POC分布初始化"""
		if self.poc_type == 'gaussian':
			self.initialize_gauss_poc(self)
		elif self.poc_type == 'uniform':
			self.initialize_uniform_poc(self)
		elif self.poc_type == '01':
			self.initialize_01_poc(self)
		else:
			raise ValueError(f"Invalid poc_type:'{self.poc_type}'")

'''
	def initialize_gauss_poc(self):
		"""高斯分布POC"""
		x = np.arange(0, self.grid_size)
		y = np.arange(0, self.grid_size)
		X, Y = np.meshgrid(x, y)
		exponent = ((X - self.gauss_center[1]) ** 2 +
					(Y - self.gauss_center[0]) ** 2) / (2 * self.sigma ** 2)
		self.poc = np.exp(-exponent)
		self.poc = (self.poc - self.poc.min()) / (self.poc.max() - self.poc.min())

	def initialize_uniform_poc(self):
		"""均匀分布POC"""
		uniform_value = 1 / (self.grid_size ** 2) * 100
		self.poc = np.full((self.grid_size, self.grid_size), uniform_value)

	def initialize_01_poc(self):
		"""01分布POC"""
		self.poc = np.zeros((self.grid_size, self.grid_size))
		if 0 <= self.target_pos_01[0] < self.grid_size and 0 <= self.target_pos_01[1] < self.grid_size:
			self.poc[self.target_pos_01] = 1
'''
