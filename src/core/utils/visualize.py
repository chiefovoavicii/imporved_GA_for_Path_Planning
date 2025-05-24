'''
可视化函数
'''

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class Visualize:
	"""可视化函数"""

	def visualize(self, env, obs, gp, paths, config):  # 多智能体路径绘制
		"""多智能体可视化（兼容单条路径和多条路径）"""
		plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
		plt.rcParams['axes.unicode_minus'] = False

		# 创建图形和坐标轴
		fig = plt.figure(figsize=(9, 6))
		ax = fig.gca()

		# 获取颜色循环
		prop_cycle = plt.rcParams['axes.prop_cycle']
		colors = prop_cycle.by_key()['color']

		# ========== 基础图层绘制（与之前相同） ==========
		# 目标概率分布热力图
		poc_matrix = gp.poc.T
		fig_poc = ax.pcolormesh(poc_matrix, cmap='Reds', alpha=0.3)
		cax1 = fig.add_axes([0.82, 0.11, 0.02, 0.77])  # [左, 下, 宽, 高]
		fig.colorbar(fig_poc, ax=ax, cax=cax1, label="目标分布概率")

		# 累计探测概率热力图
		# pos_matrix = gp.pos.T
		# fig_pos = ax.pcolormesh(pos_matrix, cmap='Greens', alpha=0.3)
		# cax2 = fig.add_axes([0.9, 0.11, 0.02, 0.77])
		# fig.colorbar(fig_pos, ax=ax, cax=cax2, label="累计探测概率")

		# 障碍物绘制
		obstacle_coords = np.argwhere(obs.obstacles == 1)
		for row, col in obstacle_coords:
			ax.add_patch(plt.Rectangle((row, col), 1, 1, facecolor='black'))

		# ========== 路径结构兼容性处理 ==========
		# 自动检测路径的嵌套层级，统一转换为列表的列表
		processed_paths = []
		for item in paths:
			# 情况1：当前项是坐标点（如 (x,y)），说明 paths 是单条路径
			if isinstance(item, tuple) and len(item) == 2:
				processed_paths = [paths]  # 将单条路径包装成列表的列表
				break
			# 情况2：当前项是路径列表（如 [(x1,y1), (x2,y2), ...]）
			elif isinstance(item, (list, tuple)):
				processed_paths.append(item)
		# 如果没有处理（例如 paths 是空列表），保持原样
		if not processed_paths:
			processed_paths = paths

		# ========== 多路径绘制（使用处理后的路径） ==========
		for i, path in enumerate(processed_paths):
			color = colors[i % len(colors)]

			# 提取路径中的坐标点
			try:
				# 检测路径中的每个点是否为二元组
				path_points = []
				for point in path:
					if isinstance(point, (list, tuple)) and len(point) == 2:
						path_points.append((point[0], point[1]))
					else:
						raise ValueError(f"路径 {i} 中存在无效坐标点: {point}")
			except TypeError:
				# 如果 path 不是可迭代对象（例如整型），直接报错
				raise TypeError(f"路径 {i} 必须是包含坐标点的列表或元组，实际类型: {type(path)}")

			# 绘制路径线条
			path_x = [x + 0.5 for (x, y) in path_points]
			path_y = [y + 0.5 for (x, y) in path_points]
			ax.plot(path_x, path_y, color=color, linewidth=2.5, alpha=0.8,
					label=f'智能体 {i + 1} 路径')

			# 绘制起点
			if len(path_points) >= 1:
				start_x, start_y = path_points[0]
				ax.scatter(start_x + 0.5, start_y + 0.5,
					color=color, marker='s', s=50, edgecolor='black', zorder=4)
			# 绘制终点
			if len(path_points) >= 1:
				end_x, end_y = path_points[-1]
				ax.scatter(end_x + 0.5, end_y + 0.5,
					color=color, marker='o', s=50, edgecolor='black', linewidth=1.2, zorder=4)

		# ========== 特殊标记 ==========
			# 目标区域
		if gp.poc_type in ('gaussian', '01'):
			max_pos_coords = np.unravel_index(np.argmax(gp.poc), gp.poc.shape)
			ax.scatter(max_pos_coords[0] + 0.5, max_pos_coords[1] + 0.5,
					   c='gold', marker='*', s=200, edgecolor='maroon',
					   linewidth=1, label='目标区域', zorder=5)

		# ========== 坐标轴美化 ==========
		ax.set_xticks(np.arange(0, env.grid_length, 1), minor=False)
		ax.set_yticks(np.arange(0, env.grid_width, 1), minor=False)
		ax.grid(which='major', color='gray', linestyle='--', linewidth=0.5)

		# ========== 自定义图例（加入起点和终点） ==========
		# 获取当前自动生成的图例句柄和标签（路径、目标区域）
		handles, labels = ax.get_legend_handles_labels()
		# 创建自定义图例句柄：起点（方块）和终点（圆点）
		start_legend = Line2D(
			[], [],
			marker='s',  # 方块标记
			markersize=10,
			markerfacecolor='white',
			markeredgecolor='black',
			markeredgewidth=1.2,
			linestyle='None',  # 不显示线条
			label='起点'
		)
		end_legend = Line2D(
			[], [],
			marker='o',  # 圆点标记
			markersize=10,
			markerfacecolor='white',
			markeredgecolor='black',
			markeredgewidth=1.2,
			linestyle='None',
			label='终点'
		)

		# 合并图例句柄和标签
		handles = handles + [start_legend, end_legend]
		labels = labels + ['起点', '终点']
		# 绘制图例
		ax.legend(
			handles=handles,
			labels=labels,
			loc='center left',
			bbox_to_anchor=(0, 0.5),
			borderaxespad=-13,
			fontsize=10,
			framealpha=0.9
		)
		#ax.legend(loc='upper left', bbox_to_anchor=(0, 0.5), borderaxespad=-13)
		plt.subplots_adjust(right=0.7, left=0.3) # 左右两边各空出百分之几的空间
		if config["poc_type"] == "dynamic_gaussian":
			ax.plot([0, 15], [2.5, 2.5], 'r--', label='目标轨迹')

		plt.show()

	def visualize_cdp_and_reward(self, cdp_list, reward_list):
		# 创建画布和子图 (1行2列)
		fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  # 调整画布宽度

		# --- 子图1：CDP曲线 ---
		ax1.plot(
			range(len(cdp_list)),
			cdp_list,
			color='#1f77b4',  # 蓝色系
			linewidth=1.5,
			marker='o',
			markersize=3,
			label='CDP'
		)
		ax1.set_title("Cumulative Detection Probability (CDP)", fontsize=12)
		ax1.set_xlabel("Episode", fontsize=10)
		ax1.set_ylabel("CDP Value", fontsize=10, color='#1f77b4')
		ax1.tick_params(axis='y', labelcolor='#1f77b4')
		ax1.grid(True, linestyle='--', alpha=0.6)
		ax1.set_xlim(0, len(cdp_list))  # 与横轴对齐
		ax1.set_ylim(bottom=0)  # 从0开始

		# --- 子图2：Reward曲线 ---
		ax2.plot(
			range(len(reward_list)),
			reward_list,
			color='#d62728',  # 红色系
			linewidth=1.5,
			marker='s',  # 改用方块标记，增强区分度
			markersize=3,
			label='Total Reward'
		)
		ax2.set_title("Total Reward", fontsize=12)
		ax2.set_xlabel("Episode", fontsize=10)
		ax2.set_ylabel("Reward", fontsize=10, color='#d62728')
		ax2.tick_params(axis='y', labelcolor='#d62728')
		ax2.grid(True, linestyle='--', alpha=0.6)
		ax2.set_xlim(0, len(reward_list))
		ax2.set_ylim(bottom=0)

		# 自动调整子图间距
		plt.tight_layout(pad=3.0)  # pad控制子图间距

		#plt.show()

		# 保存图片（可选）
		# plt.savefig("cdp_reward_subplots.png", dpi=300, bbox_inches='tight')
