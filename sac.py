from stable_baselines3 import SAC
from dwa_obstacles_env import DwaEnv
from stable_baselines3.common.utils import get_schedule_fn


# 创建环境
env = DwaEnv()

# 创建 SAC 模型
model = SAC(
    "MlpPolicy",
    env,
    buffer_size=100_000,
    batch_size=256,
    tau=0.01,
    gamma=0.98,
    train_freq=(64, "step"),
    gradient_steps=32,
    ent_coef="auto_0.5",
    learning_starts=200,
    policy_kwargs=dict(net_arch=[256, 256]),
    verbose=1,
    tensorboard_log="./tensorboard/"  # ✅ 正确写法
)

# 开始训练
model.learn(
    total_timesteps=300_000,
    log_interval=1,
    tb_log_name="SAC"  # ✅ 设置日志子目录名
)

# 保存模型
model.save("sac_dwa_model_3")
