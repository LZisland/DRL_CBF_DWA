始终显示详情

复制
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义障碍物中心和半径
obstacles = np.array([
    [2.0, 2.0, 2.0],
    [2.0, 8.0, 8.0],
    [8.0, 2.0, 8.0],
    [8.0, 8.0, 2.0],
])
radius = 1.0

# 假设给定的起点和终点
start_point = np.array([1.0, 1.0, 1.0])
goal_point  = np.array([9.0, 9.0, 9.0])

# 球面参数
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
u, v = np.meshgrid(u, v)

# 创建 3D 图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# 绘制每个障碍物为半透明球体
for center in obstacles:
    x = center[0] + radius * np.cos(u) * np.sin(v)
    y = center[1] + radius * np.sin(u) * np.sin(v)
    z = center[2] + radius * np.cos(v)
    ax.plot_surface(x, y, z, alpha=0.3)

# 绘制障碍物中心点
ax.scatter(obstacles[:, 0], obstacles[:, 1], obstacles[:, 2], color='grey', s=50, label='Obstacles')

# 绘制起点和终点
ax.scatter(*start_point, color='red',   marker='o', s=100, label='Start')
ax.scatter(*goal_point,  color='green', marker='^', s=100, label='Goal')

# 设置坐标轴范围和标签
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_zlim(0, 10)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加图例和标题
ax.legend()
ax.set_title('三维环境中的障碍物与起点/终点')

plt.show()