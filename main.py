import numpy as np
import math
from multi_obstacle_cbf import DWA
from quadrotormodel import KinematicModel
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time  # 导入time模块

matplotlib.use('TkAgg')
fig = plt.figure(1)
ax1 = fig.add_subplot(1, 1, 1, projection='3d')
# ax2 = fig.add_subplot(2, 1,2)
# ax3 = fig.add_subplot(2, 2, 4)

# 初始状态
x = np.array([0, 0, 0, 0.0, 0.0, 0.0, 0.0])
dwa = DWA()
goal = dwa.target
trajectory = np.array(x)
min_dist = float('inf')          # ← 1) 全局最小距离
trajectory_end = np.array(x)
ob = dwa.ob
v1, h1, vel1, c1, sum = [], [], [], [], []
step_counter = 0      # 记录已经走过的步长

while True:

    u, predicted_trajectory, h, v, c = dwa.dwa_control(x, goal, ob)
    # u, predicted_trajectory = dwa.dwa_control(x, goal, ob)
    x = KinematicModel(x, u, dwa.dt)
    trajectory = np.vstack((trajectory, x))
    # trajectory_end = np.vstack((trajectory_end, predicted_trajectory(-1, )))

    cur_dist = dwa.dist(x, ob)
    if cur_dist < min_dist:
        min_dist = cur_dist

    plt.cla()
    # 预测轨迹绘制
    if predicted_trajectory.ndim == 2 and predicted_trajectory.shape[0] > 1:
        ax1.plot(predicted_trajectory[:, 0], predicted_trajectory[:, 1], predicted_trajectory[:, 2], color="green",
                 linestyle="-", linewidth=1)
    ax1.scatter(x[0], x[1], x[2], "xr")
    # 设置地图的大小
    map_size = 13
    # 生成空地图
    map_data = np.zeros((map_size, map_size, map_size))
    # 将所有生成的点转换成 np.array
    ob = np.array(ob)
    # 设置起始点和终点
    start = (0, 0, 0)
    end = dwa.target
    # 定义多个球形障碍物
    obstacles = [
        {"center": ob[0], "radius": 2},
    ]

    # 创建球体表面参数网格
    u, v = np.mgrid[0:2 * np.pi:40j, 0:np.pi:20j]

    # 绘制每个障碍球
    for obs in obstacles:
        cx, cy, cz = obs["center"]
        r = obs["radius"]
        xs = cx + r * np.cos(u) * np.sin(v)
        ys = cy + r * np.sin(u) * np.sin(v)
        zs = cz + r * np.cos(v)
        ax1.plot_surface(xs, ys, zs, alpha=0.4, color="gray", edgecolor='black', linewidth=0.2)

    # 绘制起始点和终点
    ax1.scatter(start[0], start[1], start[2], color='g', s=100, marker='D', edgecolor='k', label="Start")
    ax1.scatter(end[0], end[1], end[2], color='b', s=100, marker='*', edgecolor='k', label="End")
    # 设置坐标轴标签
    ax1.set_xlabel('X [m]')
    ax1.set_ylabel('Y [m]')
    ax1.set_zlabel('Z [m]')
    ax1.set_box_aspect([1, 1, 1])
    plt.pause(0.001)
    # 检查是否到达
    dist_to_goal = math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2 + (x[2] - goal[2]) ** 2)
    # print(dist_to_goal)
    step_counter += 1
    if dwa.dist(x, goal.reshape(1, -1)) <= 0.3:
        print(f"到达目标，整个过程中最近距离 = {min_dist:.3f} m")
        break

    if dwa.counter > 500:
        print(f"超时终止，整个过程中最近距离 = {min_dist:.3f} m")
        break
print("Done")
ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "-r")

# # 定义列名（根据状态变量含义命名）
# column_names = ['position_x', 'position_y', 'position_z',
#                 'velocity_xoy', 'velocity_z', 'velocity_w','position_w']  # 根据实际状态变量调整名称
#
# # 创建DataFrame并保存到Excel
# df = pd.DataFrame(trajectory, columns=column_names)
# timestamp = time.strftime("%Y%m%d-%H%M%S")  # 生成时间戳防止覆盖
# filename = f"cbf_dwa.xlsx"
# df.to_excel(filename, index=False)
# print(f"轨迹数据已保存到: {filename}")

plt.show()