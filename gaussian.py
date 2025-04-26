import numpy as np

def initialize_gauss_poc(obj):
	"""高斯分布POC"""
	x = np.arange(0, obj.env.grid_size)
	y = np.arange(0, obj.env.grid_size)
	X, Y = np.meshgrid(x, y)
	exponent = ((X - obj.gauss_center[1]) ** 2 +
				(Y - obj.gauss_center[0]) ** 2) / (2 * obj.sigma ** 2)
	obj.poc = np.exp(-exponent)
	obj.poc = (obj.poc - obj.poc.min()) / (obj.poc.max() - obj.poc.min())