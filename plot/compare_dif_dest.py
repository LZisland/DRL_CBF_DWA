import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 设置字体为 Times New Roman，符合科研绘图标准
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "text.usetex": False
})
matplotlib.use('TkAgg')

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

# 读取表格数据
file_path1 = "cbf_dwa_0.5_2.xlsx"
file_path2 = "cbf_dwa_1_2.xlsx"
file_path3 = "cbf_dwa_5_2.xlsx"
file_path4 = "cbf_dwa_10_2.xlsx"
file_path5 = "cbf_dwa_15_2.xlsx"

df1 = pd.read_excel(file_path1)
df2 = pd.read_excel(file_path2)
df3 = pd.read_excel(file_path3)
df4 = pd.read_excel(file_path4)
df5 = pd.read_excel(file_path5)

# 提取轨迹中的位置坐标
x1, y1, z1 = df1["position_x"], df1["position_y"], df1["position_z"]
x2, y2, z2 = df2["position_x"], df2["position_y"], df2["position_z"]
x3, y3, z3 = df3["position_x"], df3["position_y"], df3["position_z"]
x4, y4, z4 = df4["position_x"], df4["position_y"], df4["position_z"]
x5, y5, z5 = df5["position_x"], df5["position_y"], df5["position_z"]

# 起点和终点
start = (0, 0, 0)
end = (10, 10, 10)

# 障碍物定义
obstacles = [[5, 5, 5]]
r = 2  # 障碍物半径

# 开始绘图
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")  # 坐标轴背景白色
fig.patch.set_facecolor('white')  # 整体图背景白色
# 设置3D坐标轴背景为白色
ax.xaxis.pane.set_facecolor('white')
ax.yaxis.pane.set_facecolor('white')
ax.zaxis.pane.set_facecolor('white')

# 绘制三条轨迹
ax.plot(x1, y1, z1, color='#377EB8', alpha=1, linewidth=1)
ax.plot(x2, y2, z2, color='#377EB8', alpha=1, linewidth=1)
ax.plot(x3, y3, z3, color='#377EB8', alpha=1, linewidth=1)
ax.plot(x4, y4, z4, color='#377EB8', alpha=1, linewidth=1)
ax.plot(x5, y5, z5, color='#377EB8', alpha=1, linewidth=1)

# 定义多个障碍物及其半径（共心不同半径）
# 定义多个共心障碍物：半径由大到小，透明度由低到高
obstacles = [
    {"center": [5, 5, 5], "radius": 2,   "alpha": 0.2}
]

# 绘制障碍物球体
u = np.linspace(0, 2 * np.pi, 30)
v = np.linspace(0, np.pi, 30)
for obs in obstacles:
    ox, oy, oz = obs["center"]
    r = obs["radius"]
    alpha = obs["alpha"]
    x_sphere = r * np.outer(np.cos(u), np.sin(v)) + ox
    y_sphere = r * np.outer(np.sin(u), np.sin(v)) + oy
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v)) + oz
    ax.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=alpha, linewidth=0)

# 示例：从 [6,5,5] 指向 [5,5,5]，即 XOY 平面中指向球心
draw_arrow_with_cone(ax, start=[5 + 2.5*np.sqrt(2)/2, 5 - 2.5*np.sqrt(2)/2, 5], end=[5, 5, 5],
                     cone_length=0.3, cone_radius=0.1, color='black')
# 添加标签
ax.text(5 + 2.6*np.sqrt(2)/2, 5 - 2.6*np.sqrt(2)/2, 5, r"$r=2$", fontsize=12, fontfamily='Times New Roman')



# 起点终点标记
ax.scatter(*start, marker="o", s=60, c="black")
ax.scatter(*end, marker="^", s=80, c="red")

# 坐标轴设置
ax.set_xlabel("X (m)", labelpad=4)  # 10是典型的默认间距，可调整
ax.set_ylabel("Y (m)", labelpad=4)
ax.set_zlabel("Z (m)", labelpad=4)
ax.set_zticks([0, 5, 10])
# 等比例坐标轴
ax.set_box_aspect([1, 1, 1])
ax.set_xlim(-1, 13)
ax.set_ylim(-1, 13)
ax.set_zlim(-1, 10.5)
# 图例与布局
ax.view_init(elev=65, azim=-135)

# 保存图片（高分辨率）
plt.savefig("dif_alpha_compare_plot.pdf", dpi=300)

# 显示图像
plt.show()
