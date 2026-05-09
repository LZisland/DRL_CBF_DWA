import gym as gym
import math
from gym import spaces
import numpy as np
from multi_obstacle_cbf import DWA  # 假设test.py在同一目录下
from quadrotormodel import KinematicModel
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from append_to_excel import app_to_xlsx

matplotlib.use('TkAgg')
ax1 = plt.axes(projection='3d')

class DwaEnv(gym.Env):
    def __init__(self):
        super(DwaEnv, self).__init__()
        self.dwa = DWA()
        self.target = self.dwa.target
        self.ob = self.dwa.ob.copy()

        # 缩小后的动作空间范围（优化后）
        self.action_space = spaces.Box(
            low=np.array([0.5, 0.5, 0.1], dtype=np.float32),
            high=np.array([3, 3, 0.5], dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)
        self.state = None
        self.reward_total = 0
        self.counter = 0
        self.state_list = []
        self.alpha_list = []
        self.beta_list = []
        self.gamma_list = []
        self.control = None
        self.predicted_trajectory = None
        self.prev_position = None
        self.steps_without_moving = 0
        self.time_limit_steps = 500
        self.current_steps = 0
        self.time_penalty = -80
        self.prev_distance_to_goal = None
        self.start_point = None
        self.total_speed = 0
        # 1. 从 Excel 载入起终点对
        excel_path = 'start_goal_pairs_4000.xlsx'
        df = pd.read_excel(excel_path)
        starts = df[['start_x', 'start_y', 'start_z']].values
        goals = df[['goal_x', 'goal_y', 'goal_z']].values
        self.pairs = list(zip(starts, goals))  # List of (np.array, np.array)
        self.pair_idx = 0  # 下次 reset 时使用的索引
        self.reset()

    def reset(self):
        # 直接从 self.pairs 取下一对起终点
        start, goal = self.pairs[self.pair_idx]
        self.pair_idx = (self.pair_idx + 1) % len(self.pairs)  # 用完循环

        # 计算初始 yaw0
        vec_xy = goal[:2] - start[:2]
        yaw0 = np.arctan2(vec_xy[1], vec_xy[0])

        # 将 start, yaw0 组装为初始 state
        self.state = np.array([
            start[0], start[1], start[2],
            0.0,  # 水平速度
            0.0,  # 垂直速度
            0.0,  # yaw rate
            yaw0
        ], dtype=np.float32)

        # 更新目标
        self.target = goal.astype(np.float32)
        self.dwa.target = self.target.copy()

        # 重置其它变量
        self.start_point = start.copy()
        self.prev_distance_to_goal = np.linalg.norm(start - goal)
        self.prev_position = start.copy()
        self.steps_without_moving = 0
        self.current_steps = 0
        self.reward_total = 0
        self.dwa.control_opt = [0, 0, 0]
        self.state_list = []
        self.alpha_list = []
        self.beta_list = []
        self.gamma_list = []
        self.predicted_trajectory = np.empty((0, 3))
        self.total_speed = 0
        self.counter = 0
        return self._get_obs()

    def _get_obs(self):
        dist_to_obstacle = self.dwa.dist(self.state, self.ob)
        relative_pos = self.target - self.state[:3]
        vx, vy, vz = self.state[3] * math.cos(self.state[6]), self.state[3] * math.sin(self.state[6]), self.state[4]
        yaw, yaw_rate = self.state[6], self.state[5]

        obs = np.concatenate([
            self.state[:3],
            [dist_to_obstacle],
            relative_pos,
            [vx, vy, vz],
            [yaw, yaw_rate]
        ])
        return obs.astype(np.float32)

    def step(self, action):
        info = {}
        alpha, beta, gamma = action
        self.dwa.alpha = alpha
        self.dwa.beta = beta
        self.dwa.gamma = gamma
        self.control, self.predicted_trajectory, _, _, _ = self.dwa.dwa_control(self.state, self.target, self.ob)
        self.state = KinematicModel(self.state.copy(), self.control, self.dwa.dt)
        self.counter += 1

        done = False

        reward_current = 0
        dist_to_obstacle = self.dwa.dist(self.state, self.ob)
        dist_to_goal = np.linalg.norm(self.state[:3] - self.target)
        delta_dist = self.prev_distance_to_goal - dist_to_goal
        self.prev_distance_to_goal = dist_to_goal
        velocity = np.linalg.norm(self.state[3:5])

        # 靠近目标给予奖励 + 时间惩罚
        reward_current += delta_dist * 10
        self.reward_total += reward_current

        # 终止条件处理
        if dist_to_obstacle <= self.dwa.safe_distance-0.2:
            reward_current += -80
            done = True
            print("碰撞，结束", self.reward_total - 100, )
            info["result"] = "collision"
        elif dist_to_goal <= self.dwa.quadrotor_radius:
            reward_current += 50
            done = True
            print("到达目标，结束", self.reward_total + 50)
            info["result"] = "success"
        elif velocity < 0.1:
            self.steps_without_moving += 1
            if self.steps_without_moving > 100:
                reward_current += -100
                done = True
                print("长时间未移动或移动速度过低，结束", self.reward_total - 80)
                info["result"] = "speedlow"
        else:
            self.steps_without_moving = 0

        self.current_steps += 1
        if self.current_steps >= self.time_limit_steps:
            reward_current += self.time_penalty
            done = True
            print("超时，结束", self.reward_total - self.time_penalty)
            info["result"] = "timeout"

        velocity = np.linalg.norm(self.state[3:5])
        self.total_speed += velocity
        if done:
            average_speed = self.total_speed / self.counter
            print(f"平均速度={average_speed}")
        if velocity > 0.2:
            reward_current += 1.0  # 鼓励移动
        else:
            reward_current -= 1.0  # 罚站

        reward_current += -0.5



        self.prev_position = self.state[:3].copy()
        return self._get_obs(), reward_current, done, {}

    def render(self):
        pass

    def close(self):
        pass




