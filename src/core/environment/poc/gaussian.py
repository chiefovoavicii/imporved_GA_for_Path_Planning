import numpy as np

def initialize_gauss_poc(obj):
	"""高斯分布POC"""
	x = np.arange(0, obj.env.grid_width)
	y = np.arange(0, obj.env.grid_length)
	X, Y = np.meshgrid(x, y)
	exponent = ((X - obj.gauss_center[1]) ** 2 +
				(Y - obj.gauss_center[0]) ** 2) / (2 * obj.sigma ** 2)
	obj.poc = np.exp(-exponent)
	obj.poc = (obj.poc - obj.poc.min()) / (obj.poc.max() - obj.poc.min())

def update_dynamic_gaussian_poc(obj):
	"""动态高斯分布POC"""
	x = obj.env.current_time * (obj.env.grid_length / obj.env.total_time)
	y = 2
	"""
	half_time = obj.env.total_time / 2
	if obj.env.current_time <= half_time:
		x = 50 * (obj.env.current_time / half_time)
		y = 40 - 20 * (obj.env.current_time / half_time)
	else:
		x = 50 + 50 * ((obj.env.current_time - half_time) / half_time)
		y = 20 + 20 * ((obj.env.current_time - half_time) / half_time)
	"""
	initialize_dynamic_gauss_poc(obj, x, y)

def initialize_dynamic_gauss_poc(obj, cx, cy):
	"""动态高斯分布POC核心计算"""
	x = np.arange(0, obj.env.grid_width)
	y = np.arange(0, obj.env.grid_length)
	X, Y = np.meshgrid(x, y)
	exponent = ((X - cy) ** 2 + (Y - cx) ** 2) / (2 * obj.sigma ** 2)
	obj.poc = np.exp(-exponent)   # 可适当缩放
	obj.poc /= obj.poc.max()  # 归一化

def get_current_target(obj):
	"""实时计算目标坐标"""
	return (
		int(obj.env.current_time * (obj.env.grid_length / obj.env.total_time)),
		int(2)
	)
	"""
	half_time = obj.env.total_time / 2
	if obj.env.current_time <= half_time:
		return (
			int(50 * (obj.env.current_time / half_time)),
			int(40 - 20 * (obj.env.current_time / half_time))
		)
	else:
		return (
			int(50 + 50 * ((obj.env.current_time - half_time) / half_time)),
			int(20 + 20 * ((obj.env.current_time - half_time) / half_time))
		)
	"""