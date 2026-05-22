import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import MaxNLocator, MultipleLocator, FuncFormatter
import numpy as np

# 创建坐标轴
fig, ax = plt.subplots(figsize=(10, 8))

# ==== 用户可直接修改的部分 ====
# 点坐标 (x, y)
points = [
    (0.26, 59.45),
    (0.89, 41.09),
    (0.38, 40.99),
    (0.067, 47.48),
    (0.78, 36.89),
    (1.17, 50.38),
    (0.14, 48.98),
    (9.43, 49.96),
    (0.28, 47.67),
    (0.45, 49.14)
]

# 对应半径
radii = [1, 1.3, 0.99, 0.5, 2.5, 5, 1.2, 0.8, 2, 0.76]

# 对应颜色 (RGB)
colors = [
    (233, 104, 122),       # Red
    (246, 223, 214),       # Blue
    (179, 198, 187),       # Green
    (214, 205, 190),     # Orange
    (72, 170, 191),     # Purplef
    (131, 189, 204),     # Cyan
    (127, 186, 167),     # Magenta
    (219, 226, 214),     # Yellow
    (247, 241, 216),      # Brown
    (166, 166, 166)    # Pink
]
# ===============================

# ===== 设置x轴刻度值（分成两段：0-1.8 和 9）=====
# 第一段刻度
x_ticks_left = [0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
# 第二段刻度（只有一个值9）
x_ticks_right = [8,13]

# 设置两段的显示范围（让刻度间距看起来等距）
# 第一段：0-1.8 占据左边60%的空间
# 第二段：9 占据右边40%的空间
left_portion = 0.9  # 左边部分占60%
right_portion = 0.1  # 右边部分占40%

# 计算显示位置（归一化到0-1）
# 第一段：0-1.8 映射到 0 到 left_portion
# 第二段：9 映射到 left_portion 到 1
x_display_min = 0
x_display_max = 1

# 设置x轴显示范围（归一化坐标）
ax.set_xlim(x_display_min, x_display_max)

# 计算实际刻度在显示中的位置
def get_display_pos(x_value):
    """将实际x值转换为显示位置"""
    if x_value <= 1.8:
        # 第一段：0-1.8 映射到 0 到 left_portion
        return (x_value / 1.8) * left_portion
    else:
        # 第二段：12 映射到 left_portion 到 1
        return left_portion + (x_value - 1.8) / (12 - 1.8) * right_portion

# 设置刻度位置（使用显示位置）
all_ticks = x_ticks_left + x_ticks_right
tick_positions = [get_display_pos(tick) for tick in all_ticks]
ax.set_xticks(tick_positions)
ax.set_xticklabels([str(tick) for tick in all_ticks])

# 禁用x轴自动缩放
ax.set_autoscalex_on(False)

# 设置y轴范围，根据数据自动扩展一些边距
x_values, y_values = zip(*points)
y_min, y_max = min(y_values), max(y_values)
ax.set_ylim(y_min-3, y_max+2)

# 计算x和y轴的单位长度比例，用于调整圆的绘制
# x轴范围：1.0（归一化后）
# y轴范围：y_max+2 - (y_min-3)
x_range = 1.0
y_range = (y_max + 2) - (y_min - 3)
# 获取图形尺寸（英寸）
fig_width, fig_height = fig.get_size_inches()
# 计算单位长度比例：y轴单位长度 / x轴单位长度
# 在数据坐标中，如果想让圆在视觉上保持圆形，需要调整半径
unit_ratio = (y_range / x_range) * (fig_width / fig_height)

# 画圆（需要将x坐标映射到显示位置，并根据单位长度比例调整半径）
for i in range(len(points)):
    x_orig, y = points[i]
    # 将原始x坐标映射到显示位置
    x_display = get_display_pos(x_orig)
    # 使用Ellipse而不是Circle，根据单位长度比例调整宽度和高度
    # 这样圆在视觉上会保持圆形
    # width是x方向的直径，height是y方向的直径
    ellipse = patches.Ellipse((x_display, y),
                              width=radii[i] * 2 / unit_ratio,  # x方向直径（根据比例调整）
                              height=radii[i] * 2,               # y方向直径（原始）
                              color=np.array(colors[i])/255.0, alpha=0.5)
    ax.add_patch(ellipse)

# 在中间绘制双斜线折断符号（表示跳过）
# 计算1.8和8在显示中的位置
break_start_x = get_display_pos(1.8)  # 1.8的位置
break_end_x = get_display_pos(8)     # 8的位置

break_y = ax.get_ylim()[0]  # x轴底部位置

# 计算双斜线的中心位置
break_center_x = (break_start_x + break_end_x) / 2

# 双斜线的参数设置
break_line_length = (break_end_x - break_start_x) * 0.12  # 单条斜线长度（占折断区域宽度的12%）
break_line_height = 0.015 * (y_max - y_min)  # 斜线的高度（相对于y轴范围）
break_line_spacing = (break_end_x - break_start_x) * 0.08  # 两条斜线之间的水平间距

# 绘制双斜线（两条平行的斜线，都从左上到右下，形成 // 效果）
# 第一条斜线（左侧）：从左上到右下
ax.plot([break_center_x - break_line_spacing/2 - break_line_length/2,
         break_center_x - break_line_spacing/2 + break_line_length/2],
        [break_y + break_line_height, break_y - break_line_height],
        'k-', linewidth=2.5, clip_on=False, zorder=10)

# 第二条斜线（右侧）：从左上到右下（与第一条平行）
ax.plot([break_center_x + break_line_spacing/2 - break_line_length/2,
         break_center_x + break_line_spacing/2 + break_line_length/2],
        [break_y + break_line_height, break_y - break_line_height],
        'k-', linewidth=2.5, clip_on=False, zorder=10)

# 设置坐标轴字体为 Times New Roman，加粗加大
ax.set_xlabel('Inference Time (seconds)', fontsize=25, fontweight='bold', fontname='Times New Roman')
ax.set_ylabel('Fusion Evaluation (SD)', fontsize=25, fontweight='bold', fontname='Times New Roman')

# 保持圆形比例
# ax.set_aspect('equal')  # 如果需要保持圆形比例，取消注释

# y轴保持原来的设置
ax.yaxis.set_major_locator(MaxNLocator(integer=True, prune='lower', steps=[1, 2, 5, 10]))

# 在坐标轴上显示数值
ax.tick_params(axis='both', which='major', labelsize=13, labelcolor='black', direction='inout', length=8)

# 调整布局
plt.tight_layout()

# ===== 保存图片 =====
# 保存路径和文件名（可根据需要修改）
save_path = './figure/plot_result.png'  # 保存路径，可以修改为任意路径和文件名
save_dpi = 300  # 保存分辨率（DPI），300是高质量，可以调整为100-600之间的值
save_format = 'png'  # 保存格式：'png', 'pdf', 'svg', 'eps', 'jpg' 等

# 创建保存目录（如果不存在）
import os
os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

# 保存图片
plt.savefig(save_path, dpi=save_dpi, format=save_format, bbox_inches='tight',
            facecolor='white', edgecolor='none')
print(f"图片已保存至: {save_path}")

# 显示图形
plt.show()

