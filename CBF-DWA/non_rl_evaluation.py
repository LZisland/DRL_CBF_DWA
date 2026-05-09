import numpy as np
import math
from multi_obstacle_cbf import DWA
from quadrotormodel import KinematicModel
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time  # 导入time模块

# 1. 从 Excel 载入起终点对
excel_path = 'start_goal_pairs_4000.xlsx'
df = pd.read_excel(excel_path)
starts = df[['start_x', 'start_y', 'start_z']].values
goals = df[['goal_x', 'goal_y', 'goal_z']].values
pairs = list(zip(starts, goals))  # List of (np.array, np.array)
pair_idx = 0  # 下次 reset 时使用的索引

for i in range(400):
    start, goal = pairs[pair_idx]
    pair_idx = (pair_idx + 1) % len(pairs)  # 用完循环
    # 初始状态
    x = np.array([start[0], start[1], start[2], 0.0, 0.0, 0.0, 0.0])
    dwa = DWA()
    goal = goal
    trajectory = np.array(x)
    trajectory_end = np.array(x)
    ob = dwa.ob
    v1, h1, vel1, c1, sum = [], [], [], [], []
    step_counter = 0      # 记录已经走过的步长
    step_slow = 0
    P = True
    while P:
        u, predicted_trajectory, h, v, c = dwa.dwa_control(x, goal, ob)
        # u, predicted_trajectory = dwa.dwa_control(x, goal, ob)
        x = KinematicModel(x, u, dwa.dt)
        trajectory = np.vstack((trajectory, x))
        # 检查是否到达
        dist_to_goal = math.sqrt((x[0] - goal[0]) ** 2 + (x[1] - goal[1]) ** 2 + (x[2] - goal[2]) ** 2)
        step_counter += 1
        if dist_to_goal <= 0.3:
            print(f"第{i}回合到达目标，步长 = {step_counter}")
            P = False
        d_ob_min = 100
        for j in range(len(ob)):
            d_ob = np.linalg.norm(x[:3] - ob[j])
            if d_ob < d_ob_min:
                d_ob_min = d_ob
        if np.sqrt(x[3]**2+x[4]**2)<0.1:
            step_slow += 1
        if step_slow > 100:
            print(f"第{i}回合移动速度过低，步长 = {step_counter}")
            P = False
        if d_ob_min < dwa.safe_distance-0.1:
            print(f"第{i}回合碰撞，步长 = {step_counter}, 最小距离{d_ob_min}")
            P = False
        if dwa.counter >= 500:
            print(f"第{i}回合超时终止，步长 = {step_counter}")
            P = False
    total_speed = 0
    for i in range(len(trajectory)):
        total_speed += np.sqrt(trajectory[i][3]**2+trajectory[i][4]**2)
    average_speed = total_speed / len(trajectory)
    print(average_speed)