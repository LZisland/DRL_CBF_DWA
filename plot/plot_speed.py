import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch

matplotlib.use('TKAgg')
# 读取数据
dwa1_df = pd.read_excel("cbf_dwa_0.5.xlsx")
dwa2_df = pd.read_excel("cbf_dwa_1.xlsx")
dwa3_df = pd.read_excel("cbf_dwa_1.5.xlsx")
dwa4_df = pd.read_excel("cbf_dwa_2.xlsx")

# 计算速度模
dwa1_speed = np.sqrt(
    dwa1_df['velocity_x']**2 +
    dwa1_df['velocity_y']**2
)
dwa2_speed = np.sqrt(
    dwa2_df['velocity_x']**2 +
    dwa2_df['velocity_y']**2
)
dwa3_speed = np.sqrt(
    dwa3_df['velocity_x']**2 +
    dwa3_df['velocity_y']**2
)
dwa4_speed = np.sqrt(
    dwa4_df['velocity_x']**2 +
    dwa4_df['velocity_y']**2
)


# 统一时间步：以最大长度为准
max_len = max(len(dwa1_speed), len(dwa2_speed), len(dwa3_speed), len(dwa4_speed))
time_steps = np.arange(max_len)

# # 补齐较短数组
# def pad_to(arr, length):
#     return np.pad(arr, (0, length - len(arr)), mode='constant', constant_values=np.nan)
#
# cbf_dwa_speed = pad_to(cbf_dwa_speed, max_len)
# dwa_speed = pad_to(dwa_speed, max_len)
# initdwa_speed = pad_to(initdwa_speed, max_len)

# 在绘制图形前（约第40行处）添加以下代码：

# 计算平均速度（忽略NaN值）
def calculate_avg_speed(speed_array):
    """计算有效速度的平均值"""
    valid_speeds = speed_array[~np.isnan(speed_array)]
    return np.mean(valid_speeds) if len(valid_speeds) > 0 else 0

# 计算各算法的平均速度
avg_speed1 = calculate_avg_speed(dwa1_speed)
avg_speed2 = calculate_avg_speed(dwa2_speed)
avg_speed3 = calculate_avg_speed(dwa3_speed)

# 打印结果
print("="*50)
print(f"{'算法':<15} | {'平均速度(m/s)':>15}")
print("-"*50)
print(f"{'DT-CBF DWA':<15} | {avg_speed1:>15.2f}")
print(f"{'Utsav DWA':<15} | {avg_speed2:>15.2f}")
print(f"{'Initial DWA':<15} | {avg_speed3:>15.2f}")
print("="*50)

# 在图形上添加标注（在plt.tight_layout()之前添加）
plt.text(0.98, 0.95,
         f"Average Speed:\n"
         f"DT-CBF: {avg_speed1:.2f} m/s\n"
         f"Utsav: {avg_speed2:.2f} m/s\n"
         f"Initial: {avg_speed3:.2f} m/s",
         transform=plt.gca().transAxes,
         fontsize=12,
         horizontalalignment='right',
         verticalalignment='top',
         bbox=dict(facecolor='white', alpha=0.8))


# 绘图
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
def mark_end(x, y, label, color):
    idx = np.where(~np.isnan(y))[0][-1]  # 最后非NaN索引
    plt.scatter(x[idx], y[idx], s=50, color=color, edgecolors='k', zorder=5)

mark_end(time_steps, dwa1_speed, '0.5', 'tab:red')
mark_end(time_steps, dwa2_speed, '1', 'tab:blue')
mark_end(time_steps, dwa3_speed, '1.5', 'tab:green')
mark_end(time_steps, dwa4_speed, '2', 'tab:orange')

plt.plot(dwa1_speed, label='$\delta$=0.5', color='tab:red', linestyle=':', linewidth=2)
plt.plot(dwa2_speed, label='$\delta$=1', color='tab:blue', linestyle='-', linewidth=2)
plt.plot(dwa3_speed, label='$\delta$=1.5', color='tab:green', linestyle='--', linewidth=2)
plt.plot(dwa4_speed, label='$\delta$=2', color='tab:orange', linestyle='-.', linewidth=2)


plt.xlim(left=0, right=170)          # X 轴：0 到最大步长-1
plt.ylim(bottom=0, top=2)          # Y 轴：0 到最大速度的 110%

plt.xlabel('t (s)', fontsize=16)
plt.ylabel('Speed (m/s)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(loc='upper right', fontsize=16, frameon=True)
plt.tight_layout()
# 调整坐标轴刻度数字的字体大小 (新增代码)
plt.tick_params(axis='both', which='major', labelsize=16)  # 主刻度
ticks = np.arange(0, 176, 25)  # 和你现在显示的步长一致
labels = [str(int(t / 10)) for t in ticks]
plt.xticks(ticks, labels)

axins = inset_axes(ax, width="40%", height="30%", loc='lower left',
                   bbox_to_anchor=(0.1, 0.1, 1, 1),
                   bbox_transform=ax.transAxes)
axins.plot(dwa1_speed, label='$\delta$=0.5', color='tab:red', linestyle=':', linewidth=2)
axins.plot(dwa2_speed, label='$\delta$=1', color='tab:blue', linestyle='-', linewidth=2)
axins.plot(dwa3_speed, label='$\delta$=1.5', color='tab:green', linestyle='--', linewidth=2)
axins.plot(dwa4_speed, label='$\delta$=2', color='tab:orange', linestyle='-.', linewidth=2)

# 设置放大区间
zone_left = 20
zone_right = 80

# 坐标轴的扩展比例（根据实际数据调整）
x_ratio = 0  # x轴显示范围的扩展比例
y_ratio = 0.05  # y轴显示范围的扩展比例

# X轴的显示范围
xlim0 = time_steps[zone_left]-(time_steps[zone_right]-time_steps[zone_left])*x_ratio
xlim1 = time_steps[zone_right]+(time_steps[zone_right]-time_steps[zone_left])*x_ratio

# Y轴的显示范围
y = np.hstack((dwa1_speed[zone_left:zone_right], dwa2_speed[zone_left:zone_right],
               dwa3_speed[zone_left:zone_right], dwa4_speed[zone_left:zone_right]))
ylim0 = np.min(y)-(np.max(y)-np.min(y))*y_ratio
ylim1 = np.max(y)+(np.max(y)-np.min(y))*y_ratio

# 调整子坐标系的显示范围
axins.set_xlim(xlim0, xlim1)
axins.set_ylim(ylim0, ylim1)

# 原图中画方框
tx0 = xlim0
tx1 = xlim1
ty0 = ylim0
ty1 = ylim1
sx = [tx0,tx1,tx1,tx0,tx0]
sy = [ty0,ty0,ty1,ty1,ty0]
ax.plot(sx,sy,"black")

# 画两条线
xy = (xlim0,ylim0)
xy2 = (xlim0,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)

xy = (xlim1,ylim0)
xy2 = (xlim1,ylim1)
con = ConnectionPatch(xyA=xy2,xyB=xy,coordsA="data",coordsB="data",
        axesA=axins,axesB=ax)
axins.add_artist(con)



# 保存为PDF
plt.savefig("fig3_speed.pdf", format='pdf', dpi=300, bbox_inches='tight')
plt.show()