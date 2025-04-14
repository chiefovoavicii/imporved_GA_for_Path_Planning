import numpy as np







def create_target_probability_map(points, x_range, y_range, grid_size, sigma=1.0):
    """
    生成二维目标概率分布图
    
    参数：
    points: 目标出现过的坐标列表，格式[(x1,y1), (x2,y2), ...]
    x_range: X轴范围，格式(x_min, x_max)
    y_range: Y轴范围，格式(y_min, y_max)
    grid_size: 网格尺寸，格式(x_resolution, y_resolution)
    sigma: 高斯核标准差，控制概率扩散范围
    
    返回：
    二维概率矩阵，形状为(grid_size[0], grid_size[1])
    """
    
    # 生成网格坐标系统
    x = np.linspace(x_range[0], x_range[1], grid_size[0])
    y = np.linspace(y_range[0], y_range[1], grid_size[1])
    xx, yy = np.meshgrid(x, y, indexing='ij')
    
    # 初始化概率图
    prob_map = np.zeros_like(xx)
    
    for (px, py) in points:
        # 计算网格点到当前点的距离平方
        dx = xx - px
        dy = yy - py
        distance_sq = dx**2 + dy**2
        
        # 生成未归一化的高斯分布
        gaussian = np.exp(-distance_sq / (2 * sigma**2))
        
        # 对当前高斯分布进行归一化（使区域积分为1）
        gaussian_normalized = gaussian / gaussian.sum()
        
        # 累加到概率图中
        prob_map += gaussian_normalized
    
    # 处理无先验信息的特殊情况
    if np.allclose(prob_map, 0):
        return np.ones_like(prob_map) / prob_map.size
    
    # 全局归一化概率图
    return prob_map / prob_map.sum()

# 示例用法
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 定义参数
    x_range = (0, 20)      # X轴范围
    y_range = (0, 20)      # Y轴范围
    grid_size = (100, 100) # 网格分辨率
    sigma = 2            # 高斯核标准差
    
    # 示例目标位置（包含轨迹点）
    points = [
        (3, 3),            # 单独位置
        (7, 7),            # 另一个单独位置
        (5, 5), (5, 6), (6, 6)  # 轨迹点
    ]
    
    # 生成概率图
    prob_map = create_target_probability_map(points, x_range, y_range, grid_size, sigma)
    
    # 可视化
    plt.figure(figsize=(10, 8))
    plt.imshow(prob_map.T, 
               origin='lower', 
               extent=[x_range[0], x_range[1], y_range[0], y_range[1]],
               cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.scatter(*zip(*points), color='red', s=10, label='Target Locations')
    plt.title('Target Probability Distribution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.show()