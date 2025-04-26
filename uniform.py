import numpy as np

def initialize_uniform_poc(obj):
	"""均匀分布POC"""
	uniform_value = 1 / (obj.env.grid_size ** 2) * 100
	obj.poc = np.full((obj.env.grid_size, obj.env.grid_size), uniform_value)