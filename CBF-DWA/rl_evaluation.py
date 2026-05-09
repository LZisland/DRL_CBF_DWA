import gym
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy

# ✅ 导入你的环境（请替换为你自己的环境）
from dwa_obstacles_env import DwaEnv  # 修改为你的env文件名和类名

# ✅ 初始化环境
env = DwaEnv()  # 如果需要图形显示，可设为True

# ✅ 加载 SAC 模型
model_path = "sac_dwa_model_3.zip"
model = SAC.load(model_path, env=env)

# # ✅ 评估模型（返回平均奖励和标准差）
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=50, render=False, deterministic=True)
# print(f"评估回合数：50")
# print(f"平均奖励：{mean_reward:.2f} ± {std_reward:.2f}")

# ✅ 如需逐回合评估并打印细节：
success_count = 0
collision_count = 0
timeout_count = 0
speedlow_count = 0
step_total = 0

for i in range(400):
    obs = env.reset()
    done = False
    total_reward = 0
    step = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1
        # env.render()  # 如需渲染取消注释

    # ✅ 根据 info 判断结果（需你环境中定义，例如 info["result"] = "success"/"collision"/"timeout"）
    result = info.get("result", "")
    if result == "success":
        success_count += 1
        step_total += step
    elif result == "collision":
        collision_count += 1
    elif result == "timeout":
        timeout_count += 1
    elif result == "speedlow":
        speedlow_count += 1

    print(f"第 {i+1} 回合：总奖励 = {total_reward:.2f}，结果 = {result}，步数 = {step}")
print(f"平均步长 = {step_total/success_count:.2f}")

print(f"\n成功次数：{success_count}")
print(f"碰撞次数：{collision_count}")
print(f"超时次数：{timeout_count}")
print(f"低速次数：{speedlow_count}")
