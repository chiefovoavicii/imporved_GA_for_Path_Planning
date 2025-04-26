import numpy as np

def initialize_01_poc(obj):
	"""01分布POC"""
	obj.poc = np.zeros((obj.env.grid_size, obj.env.grid_size))
	if 0 <= obj.target_pos_01[0] < obj.env.grid_size and 0 <= obj.target_pos_01[1] < obj.env.grid_size:
		obj.poc[obj.target_pos_01] = 1