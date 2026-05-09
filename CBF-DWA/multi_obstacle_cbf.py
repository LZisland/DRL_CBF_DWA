import math
import numpy as np
from quadrotormodel import KinematicModel


# 创建一个类
class DWA:
    def __init__(self):
        # 线速度边界
        self.vxoy_max = math.sqrt(1 ** 2 + 1 ** 2)
        self.vz_max = 1
        self.vxoy_min = -math.sqrt(1 ** 2 + 1 ** 2)
        self.vz_min = -1
        # 角速度边界
        self.vw_max = 40.0 * math.pi / 180.0  # [rad/s]
        self.vw_min = -40.0 * math.pi / 180.0  # [rad/s]
        # 线加速度和角加速度最大值
        self.axoy_max = math.sqrt(0.5 ** 2 + 0.5 ** 2)  # [m/ss]
        self.az_max = 0.5
        self.aw_max = 40.0 * math.pi / 180.0  # [rad/ss]
        # 采样分辨率
        self.vxoy_sample = 0.01
        self.vz_sample = 0.01
        self.vw_sample = 0.05
        # 离散时间间隔
        self.dt = 0.1
        # 轨迹推算时间长度
        self.predict_time = 1.5
        # # 轨迹评价函数系数
        self.alpha = 100
        self.beta = 1
        self.gamma = 0.01
        # self.alpha = []
        # self.beta = []
        # self.gamma = []
        # 与障碍物距离阀值
        self.safe_distance = 2
        # 无人机半径
        self.quadrotor_radius = 0.3
        # # 障碍物位置
        # self.ob = np.array([[3.0, 3.0, 5.0],
        #     [5.0, 2.0, 2.0],
        #     [2.0, 5.0, 3.0]])
        self.ob = np.array([
            (5,5,5)
                ])

        # 目标点位置
        self.target = np.array([10, 10, 10])
        self.control_opt = [0, 0, 0]
        self.counter = 0
        self.delta = 0.5

    # 滚动窗口算法
    def dwa_control(self, state, goal, obstacle):
        control, trajectory, h, v, c = self.trajectory_evaluation(state, goal, obstacle)
        return control, trajectory, h, v, c

    # 速度采样
    def cal_dynamic_window_vel(self, vxoy, vz, vw):
        Vm = self.cal_vel_limit()
        Vd = self.cal_accel_limit(vxoy, vz, vw)
        a = max([Vm[0], Vd[0]])
        b = min([Vm[1], Vd[1]])
        c = max([Vm[2], Vd[2]])
        d = min([Vm[3], Vd[3]])
        e = max([Vm[4], Vd[4]])
        f = min([Vm[5], Vd[5]])
        return [a, b, c, d, e, f]

    # 计算速度边界限制
    def cal_vel_limit(self):
        return [self.vxoy_min, self.vxoy_max, self.vz_min, self.vz_max, self.vw_min, self.vw_max]

    # 计算加速度限制
    def cal_accel_limit(self, vxoy, vz, vw):
        vxoy_low = vxoy - self.axoy_max * self.dt
        vxoy_high = vxoy + self.axoy_max * self.dt
        vz_low = vz - self.az_max * self.dt
        vz_high = vz + self.az_max * self.dt
        vw_low = vw - self.aw_max * self.dt
        vw_high = vw + self.aw_max * self.dt
        return [vxoy_low, vxoy_high, vz_low, vz_high, vw_low, vw_high]

    # 轨迹推算
    def trajectory_predict(self, state_init, vxoy, vz, vw):
        state = np.array(state_init)
        trajectory = state
        time = 0
        # 在预测时间段内
        while time <= self.predict_time:
            x = KinematicModel(state, [vxoy, vz, vw], self.dt)  # 运动学模型
            trajectory = np.vstack((trajectory, x))
            time += self.dt

        return trajectory

    # 轨迹评价函数
    def trajectory_evaluation(self, state, goal, obstacle):
        heading_eval, vel_eval, cbf_eval = 0, 0, 0  # 初始值
        G_max = -float('inf')  # 最优评价
        trajectory_opt = state  # 最优轨迹
        control_opt = [state[3], state[4], state[5]]  # 最优控制
        # 计算速度空间
        dynamic_window_vel = self.cal_dynamic_window_vel(state[3], state[4], state[5])
        self.counter += 1
        sum_heading, sum_vel, sum_cbf = 1,1,1  # 归一化
        # 在速度空间中按照预先设定的分辨率采样
        for vxoy in np.linspace(dynamic_window_vel[0], dynamic_window_vel[1],
                              int((dynamic_window_vel[1] - dynamic_window_vel[0]) / self.vxoy_sample)):
            for vz in np.linspace(dynamic_window_vel[2], dynamic_window_vel[3],
                                  int((dynamic_window_vel[3] - dynamic_window_vel[2]) / self.vz_sample)):
                for vw in np.linspace(dynamic_window_vel[4], dynamic_window_vel[5],
                                      int((dynamic_window_vel[5] - dynamic_window_vel[4]) / self.vw_sample)):
                        trajectory = self.trajectory_predict(state, vxoy, vz, vw)
                        # 轨迹推算
                        heading_eval = self.alpha * self.heading_judge(trajectory, goal) / sum_heading
                        vel_eval = self.beta * self.velocity_judge(trajectory) / sum_vel
                        cbf_eval = self.gamma * self.cbf_simple(trajectory, obstacle) / sum_cbf
                        # # 轨迹评价
                        if cbf_eval >= 0:
                            G = heading_eval + vel_eval + cbf_eval
                            if G_max <= G:
                                G_max = G
                                trajectory_opt = trajectory
                                control_opt = [vxoy, vz, vw]
                                self.control_opt = control_opt
                        else:
                            self.control_opt = [0, 0, 0]
                            continue
        return control_opt, trajectory_opt, heading_eval, vel_eval, cbf_eval

    # 计算无人机当前距离障碍物最近的欧式距离
    def dist(self, state, obstacle):
        ox = obstacle[:, 0]
        oy = obstacle[:, 1]
        oz = obstacle[:, 2]
        dx = state[0] - ox
        dy = state[1] - oy
        dz = state[2] - oz
        r = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        return np.min(r)

    # 方位角评价函数
    def heading_judge(self, trajectory, goal):
        # 获取目标位置与轨迹终点之间的坐标差
        dx = goal[0] - trajectory[-1, 0]
        dy = goal[1] - trajectory[-1, 1]
        dz = goal[2] - trajectory[-1, 2]
        # 计算目标点与轨迹终点之间的方向向量
        goal_direction = np.array([dx, dy, dz])
        # 获取轨迹终点的朝向（单位向量，假设轨迹终点的速度表示方向）
        trajectory_direction = np.array([trajectory[-1, 3] * math.cos(trajectory[-1, 6]), trajectory[-1, 3] * math.sin(trajectory[-1, 6]), trajectory[-1, 4]])
        # 归一化方向向量
        goal_direction_norm = np.linalg.norm(goal_direction)
        trajectory_direction_norm = np.linalg.norm(trajectory_direction)

        if goal_direction_norm == 0 or trajectory_direction_norm == 0:
            return math.pi  # 无法计算夹角时，返回一个较大的误差

        # goal_direction /= goal_direction_norm
        # trajectory_direction /= trajectory_direction_norm
        # 计算两向量之间的夹角余弦
        cos_angle = np.dot(goal_direction, trajectory_direction)/(trajectory_direction_norm * goal_direction_norm)
        # 由于浮点误差可能导致超出[-1, 1]的范围，因此进行裁剪
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # 计算夹角误差（弧度）
        error_angle = math.acos(cos_angle)
        # 返回夹角的误差值
        return math.pi/2 - error_angle

    # 速度评价函数
    def velocity_judge(self, trajectory):
        v = np.sqrt(trajectory[-1, 3] ** 2 + trajectory[-1, 4] ** 2)
        return v

    def cbf_simple(self, trajectory, obstacle):
        mu=0.1
        cbf_min = float('inf')

        for obs in obstacle:  # 直接遍历筛选后的障碍物
            for i in range(len(trajectory)-1):
                # 当前状态位置和速度向量
                pos = trajectory[i, 0:3]  # p(t)
                yaw = trajectory[i, 6]
                vxy = trajectory[i, 3]
                vz = trajectory[i, 4]
                vel_vec = np.array([vxy * math.cos(yaw),
                                    vxy * math.sin(yaw),
                                    vz])  # ˙p(t)

                # 障碍物位置
                obs_pos = obs  # [x_obs, y_obs, z_obs]

                # 相对位置向量
                delta_p = pos - obs_pos  # p(t) - p̄

                # 计算 h 在 t 时刻
                h_k = np.dot(delta_p, delta_p) + mu * np.dot(delta_p, vel_vec)

                # 预测下一时刻
                next_pos = trajectory[i, 0:3] + 0.1 * vel_vec
                # 这里假设速度在这一小步内不变，仍用 vel_vec
                delta_p_next = next_pos - obs_pos
                h_k1 = np.dot(delta_p_next, delta_p_next) + mu * np.dot(delta_p_next, vel_vec)

                # CBF 约束值
                cbf_val = (h_k1 - h_k) + self.delta * (h_k - self.safe_distance ** 2)
                # 注意：safe_distance 如果是“欧式距离阈值”，
                # 这里对比时要平方成 safe_distance**2

                cbf_min = min(cbf_min, cbf_val)

        return cbf_min


