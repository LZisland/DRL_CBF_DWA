import math


# 无人机运动学模型
# state:状态量，0：x，1：y，2：z，3：
# control:控制量，0：xoy平面速度分量，1：z轴速度分量，2：xoy平面角速度分量
# dt:离散时间
def KinematicModel(state, control, dt):
    state[0] += control[0] * math.cos(state[6]) * dt  # x
    state[1] += control[0] * math.sin(state[6]) * dt  # y
    state[2] += control[1] * dt  # z
    state[3] = control[0]
    state[4] = control[1]
    state[5] = control[2]
    state[6] += control[2] * dt
    return state
