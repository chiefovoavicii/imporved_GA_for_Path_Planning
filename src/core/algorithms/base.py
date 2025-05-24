import numpy as np

class BaseAlgorithm:
	"""所有算法的基类"""

	def __init__(self, env):  # 强制接收环境对象
		self.env = env  # 所有算法共享环境接口



