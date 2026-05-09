import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
matplotlib.use('TKAgg')

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 15,
    "axes.labelsize": 18,
    "axes.titlesize": 15,
    "legend.fontsize": 15,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "text.usetex": False
})

def draw_arrow_with_cone(ax, start, end, cone_length=0.3, cone_radius=0.1, color='black'):
    """
    绘制带三角锥头的箭头（含线段），箭头位于 XOY 平面
    """
    # 起点和方向
    start = np.array(start)
    end = np.array(end)
    direction = end - start
    direction_unit = direction / np.linalg.norm(direction)

    # 线段部分 = 箭头长度 - 锥头长度
    line_end = end - direction_unit * cone_length

    # 1. 画线段
    ax.plot([start[0], line_end[0]],
            [start[1], line_end[1]],
            [start[2], line_end[2]], color=color, linewidth=1.5)

    # 2. 画锥头
    z = direction_unit
    # 找两个垂直方向构建圆面
    not_z = np.array([1, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1, 0])
    x = np.cross(not_z, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)

    # 圆面上的点
    resolution = 20
    theta = np.linspace(0, 2*np.pi, resolution)
    circle_pts = cone_radius * (np.outer(np.cos(theta), x) + np.outer(np.sin(theta), y)) + line_end

    # 创建锥体面
    faces = [[end, circle_pts[i], circle_pts[(i+1)%resolution]] for i in range(resolution)]
    cone = Poly3DCollection(faces, color=color, alpha=1.0)
    ax.add_collection3d(cone)

# 读取数据
file_paths = [
    "cbf_dwa_0.5_2.xlsx",
    "cbf_dwa_0.5_2_0.01_10_0.1.xlsx",
    "cbf_dwa_0.5_2_1000_10_0.1.xlsx",
    "cbf_dwa_0.5_2_4_0.01_0.1.xlsx",
    "cbf_dwa_0.5_2_4_20_0.1.xlsx",
    "cbf_dwa_0.5_2_4_10_0.001.xlsx",
    "cbf_dwa_0.5_2_4_10_10.xlsx"
]

dataframes = [pd.read_excel(path) for path in file_paths]

# 提取坐标
trajectories = [(df["position_x"], df["position_y"], df["position_z"]) for df in dataframes]

# 设置起点和终点
start = (0, 0, 0)
end = (10, 10, 10)

# 设置障碍物参数
obstacle_center = [5, 5, 5]
obstacle_radius = 2

# 绘图 - 原始图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")  # 坐标轴背景白色
fig.patch.set_facecolor('white')  # 整体图背景白色
# 设置3D坐标轴背景为白色
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

# 绘制轨迹
colors = ['#E41A1C', '#377EB8', '#377EB8', '#4DAF4A', '#4DAF4A', '#FF7F00', '#FF7F00']
styles = ['-', '-', '--', '-', '--', '-', '--']
for (x, y, z), color, style in zip(trajectories, colors, styles):
    ax.plot(x, y, z, color=color, linewidth=2, linestyle=style)

# 绘制障碍物
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
x_sphere = obstacle_radius * np.outer(np.cos(u), np.sin(v)) + obstacle_center[0]
y_sphere = obstacle_radius * np.outer(np.sin(u), np.sin(v)) + obstacle_center[1]
z_sphere = obstacle_radius * np.outer(np.ones(np.size(u)), np.cos(v)) + obstacle_center[2]
ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.2, linewidth=0)

# 绘制起点终点
ax.scatter(*start, marker="o", s=60, c="black")
ax.scatter(*end, marker="^", s=80, c="red")

ax.set_xlabel("X [m]", labelpad=10)  # 10是典型的默认间距，可调整
ax.set_ylabel("Y [m]", labelpad=10)
ax.set_zlabel("Z [m]", labelpad=10)
ax.view_init(elev=80, azim=-90)
ax.set_zticks([0, 10, 20])
# 示例：从 [6,5,5] 指向 [5,5,5]，即 XOY 平面中指向球心
draw_arrow_with_cone(ax, start=[5 + 2.5*np.sqrt(2)/2, 5 - 2.5*np.sqrt(2)/2, 5], end=[5, 5, 5],
                     cone_length=0.3, cone_radius=0.1, color='black')
# 添加标签
ax.text(5 + 2.6*np.sqrt(2)/2, 5 - 2.6*np.sqrt(2)/2, 5, r"$r=2$", fontsize=12, fontfamily='Times New Roman')
ax.view_init(elev=65, azim=-135)

plt.tight_layout()
plt.savefig("dif_evalu_compare_plot.pdf", dpi=300)

# 绘图 - 子图（局部放大）
fig2 = plt.figure(figsize=(8, 6))
ax2 = fig2.add_subplot(111, projection='3d')
for (x, y, z), color, style in zip(trajectories, colors, styles):
    mask = (x >= 0) & (x <= 1.5) & (y >= 0) & (y <= 1.5) & (z >= 0) & (z <= 1.5)
    ax2.plot(x[mask], y[mask], z[mask], color=color, linewidth=1, linestyle=style)

ax.set_xlabel("X [m]", labelpad=10)  # 10是典型的默认间距，可调整
ax.set_ylabel("Y [m]", labelpad=10)
ax.set_zlabel("Z [m]", labelpad=12)
ax2.view_init(elev=65, azim=-135)


plt.tight_layout()
plt.savefig("dif_evalu_compare_plot_zoom.pdf", dpi=300)

