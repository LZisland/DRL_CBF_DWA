import numpy as np
import math
from multi_obstacle_cbf import DWA
from quadrotormodel import KinematicModel
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time

matplotlib.use('TkAgg')
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# 初始状态
x = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
dwa = DWA()
goal = dwa.target
trajectory = np.array(x)
ob = np.array(dwa.ob)
step_counter = 0
delta = dwa.delta
min_dist = float('inf')
safe_distance = dwa.safe_distance

# 开始主循环
while True:
    u, predicted_trajectory, h, v, c = dwa.dwa_control(x, goal, ob)
    x = KinematicModel(x, u, dwa.dt)
    trajectory = np.vstack((trajectory, x))
    step_counter += 1
    cur_dist = dwa.dist(x, ob)
    if cur_dist < min_dist:
        min_dist = cur_dist

    dist_to_goal = np.linalg.norm(x[:3] - goal)
    if dist_to_goal <= dwa.quadrotor_radius:
        print(f"到达目标，整个过程中最近距离 = {min_dist:.3f} m")
        break
    if dwa.counter > 200:
        print("can't find path")
        break

# 绘图：仅结束后绘制一次
ax1.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], "-r", label="Trajectory")

# 绘制圆形障碍物（球体）
r = dwa.safe_distance
u = np.linspace(0, 2 * np.pi, 20)
v = np.linspace(0, np.pi, 20)
for ox, oy, oz in ob:
    x_sphere = r * np.outer(np.cos(u), np.sin(v)) + ox
    y_sphere = r * np.outer(np.sin(u), np.sin(v)) + oy
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v)) + oz
    ax1.plot_surface(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.3, linewidth=0)

# 起点和终点
start = (0, 0, 0)
end = goal
ax1.scatter(start[0], start[1], start[2], color='g', s=100, marker='D', edgecolor='k', label="Start")
ax1.scatter(end[0], end[1], end[2], color='b', s=100, marker='*', edgecolor='k', label="End")

# 设置图像属性
ax1.set_xlabel('X [m]')
ax1.set_ylabel('Y [m]')
ax1.set_zlabel('Z [m]')
ax1.set_title("CBF-DWA Trajectory")
ax1.set_box_aspect([1, 1, 1])
ax1.legend()

# 保存轨迹数据
column_names = ['position_x', 'position_y', 'position_z',
                'velocity_x', 'velocity_y', 'velocity_z',
                'yaw_angle']
df = pd.DataFrame(trajectory, columns=column_names)
filename = f"cbf_dwa_{delta}.xlsx"
df.to_excel(filename, index=False)
print(f"轨迹数据已保存到: {filename}")

plt.show()
