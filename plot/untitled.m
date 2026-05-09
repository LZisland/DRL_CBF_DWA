% 设置字体和图像尺寸
figure('Position', [100, 100, 800, 600]);
hold on;
set(gca, 'FontName', 'Times New Roman', 'FontSize', 12);

% 读取轨迹数据
df1 = readtable("cbf_dwa_0.5_2.xlsx");
df2 = readtable("cbf_dwa_0.5_2_0.01_10_0.1.xlsx");
df3 = readtable("cbf_dwa_0.5_2_1000_10_0.1.xlsx");
df4 = readtable("cbf_dwa_0.5_2_4_0.01_0.1.xlsx");
df5 = readtable("cbf_dwa_0.5_2_4_20_0.1.xlsx");
df6 = readtable("cbf_dwa_0.5_2_4_10_0.001.xlsx");
df7 = readtable("cbf_dwa_0.5_2_4_10_10.xlsx");

% 绘制轨迹
plot3(df1.position_x, df1.position_y, df1.position_z, 'Color', [228 26 28]/255, 'LineWidth', 1);
plot3(df2.position_x, df2.position_y, df2.position_z, 'Color', [55 126 184]/255, 'LineWidth', 1);
plot3(df3.position_x, df3.position_y, df3.position_z, '--', 'Color', [55 126 184]/255, 'LineWidth', 1);
plot3(df4.position_x, df4.position_y, df4.position_z, 'Color', [77 175 74]/255, 'LineWidth', 1);
plot3(df5.position_x, df5.position_y, df5.position_z, '--', 'Color', [77 175 74]/255, 'LineWidth', 1);
plot3(df6.position_x, df6.position_y, df6.position_z, 'Color', [255 127 0]/255, 'LineWidth', 1);
plot3(df7.position_x, df7.position_y, df7.position_z, '--', 'Color', [255 127 0]/255, 'LineWidth', 1);

% 起点和终点
start = [0, 0, 0];
goal = [10, 10, 10];
scatter3(start(1), start(2), start(3), 60, 'k', 'filled', 'o');
scatter3(goal(1), goal(2), goal(3), 80, 'r', '^', 'filled');

% 绘制障碍物球体
[U, V] = meshgrid(linspace(0, 2*pi, 30), linspace(0, pi, 30));
r = 2;
ox = 5; oy = 5; oz = 5;
x_sphere = r * cos(U) .* sin(V) + ox;
y_sphere = r * sin(U) .* sin(V) + oy;
z_sphere = r * ones(size(U)) .* cos(V) + oz;
surf(x_sphere, y_sphere, z_sphere, 'FaceAlpha', 0.2, 'EdgeColor', 'none', 'FaceColor', [0.5 0.5 0.5]);

% 绘制箭头（示例方向：[6.77, 3.23, 5] -> [5, 5, 5]）
arrow_start = [5 + 2.5*sqrt(2)/2, 5 - 2.5*sqrt(2)/2, 5];
arrow_end = [5, 5, 5];
arrow_dir = arrow_end - arrow_start;
quiver3(arrow_start(1), arrow_start(2), arrow_start(3), ...
        arrow_dir(1), arrow_dir(2), arrow_dir(3), 0, 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);

% 添加箭头标签
text(5 + 2.6*sqrt(2)/2, 5 - 2.6*sqrt(2)/2, 5, '$r=2$', ...
     'Interpreter', 'latex', 'FontSize', 12, 'FontName', 'Times New Roman');

% 设置坐标轴
xlabel('X [m]', 'FontName', 'Times New Roman');
ylabel('Y [m]', 'FontName', 'Times New Roman');
zlabel('Z [m]', 'FontName', 'Times New Roman');
set(gca, 'ZTick', [0 5 10 15 20]);
axis equal;
view(-45,45);
grid on;

% 保存图像
saveas(gcf, 'dif_evalu_compare_plot.pdf');
