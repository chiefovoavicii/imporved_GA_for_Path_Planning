'''
生成目标存在概率分布:正态分布、均匀分布、0-1分布
'''

import numpy as np
from src.ui.config_parser import config
from src.core.environment.grid import GridEnvironment
from src.core.environment.poc.gaussian import initialize_gauss_poc
from src.core.environment.poc.uniform import initialize_uniform_poc
from src.core.environment.poc.zero_one import initialize_01_poc
from src.core.environment.poc.gaussian import update_dynamic_gaussian_poc
from src.core.environment.poc.gaussian import get_current_target



class GeneratePOC:
	def __init__(self, env: GridEnvironment):
		# 绑定需要的配置参数到实例属性
		self.env = env

		self.poc_type = config["poc_type"]
		self.poc = np.zeros((config["grid_length"], config["grid_width"]))  # 初始化poc矩阵
		self.pod = config["pod"]  # 初始化pod
		self.pos = np.zeros((config["grid_length"], config["grid_width"]))  # 初始化pos矩阵

		self.target_pos_01 = config["target_pos_01"]
		self.gauss_center = config["gauss_center"]
		self.sigma = config["sigma"]
		self.theta = config["theta"]

		self.initialize_gauss_poc = initialize_gauss_poc
		self.initialize_uniform_poc = initialize_uniform_poc
		self.initialize_01_poc = initialize_01_poc
		self.update_dynamic_gaussian_poc = update_dynamic_gaussian_poc
		self.get_current_target = get_current_target

	def initialize_poc(self):
		"""POC分布初始化"""
		if self.poc_type == 'gaussian':
			self.initialize_gauss_poc(self)
		elif self.poc_type == 'uniform':
			self.initialize_uniform_poc(self)
		elif self.poc_type == '01':
			self.initialize_01_poc(self)
		elif self.poc_type == 'dynamic_gaussian':
			self.update_dynamic_gaussian_poc(self)
		else:
			raise ValueError(f"Invalid poc_type:'{self.poc_type}'")
