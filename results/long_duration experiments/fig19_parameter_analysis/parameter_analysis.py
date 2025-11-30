import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, MultipleLocator
import matplotlib.patches as mpatches

# 设置出版质量的图表样式
plt.rcParams['font.family'] = 'Arial'  # Times New Roman is not available by default in matplotlib
plt.rcParams['font.size'] = 28
plt.rcParams['axes.linewidth'] = 2.0
plt.rcParams['lines.linewidth'] = 2.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.minor.width'] = 0.8
plt.rcParams['ytick.minor.width'] = 0.8

# 从LaTeX表格提取的数据
kappa = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 100])
ate = np.array([29.223, 29.221, 29.222, 29.224, 29.186, 29.176, 29.223, 
                29.224, 29.223, 29.193, 29.225, 29.224, 29.175, 29.223, 29.249])
cd = np.array([5.292, 5.289, 5.292, 5.292, 5.341, 5.337, 5.298, 
               5.292, 5.298, 5.348, 5.291, 5.290, 5.342, 5.298, 5.421])
pcg_iter = np.array([8.939, 8.963, 8.976, 8.979, 8.987, 8.988, 8.987, 
                     8.989, 8.993, 8.991, 8.995, 9.003, 9.010, 9.017, 9.036])

# Nature顶级期刊配色方案 - 更优雅的配色
color_ate = '#1F77B4'  # 经典蓝色
color_cd = '#FF7F0E'   # 橙色
color_pcg = '#2CA02C'  # 绿色
highlight_color = '#E74C3C'  # 红色高亮

# 创建图表
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
# how to adjust the space between two subplots
# plt.subplots_adjust(hspace=0.05)

# ========== 第一张图：ATE和CD vs κ ==========
# 创建双Y轴
ax1_cd = ax1.twinx()

# 绘制ATE（左Y轴）- 使用较小的标记
line1 = ax1.plot(kappa, ate, marker='o', linestyle='-', color=color_ate, 
                 linewidth=5, markersize=10, markerfacecolor='white', 
                 markeredgewidth=1.5, markeredgecolor=color_ate, label='ATE', zorder=3)

# 高亮κ=10的点
ax1.plot(10, ate[9], marker='o', markersize=10, color=highlight_color, 
         markerfacecolor=highlight_color, markeredgewidth=1.5, 
         markeredgecolor=color_ate, zorder=4)

# 绘制CD（右Y轴）- 使用较小的标记
line2 = ax1_cd.plot(kappa, cd, marker='s', linestyle='-', color=color_cd, 
                    linewidth=5, markersize=10, markerfacecolor='white', 
                    markeredgewidth=1.5, markeredgecolor=color_cd, label='CD', zorder=3)

# 高亮κ=10的点
ax1_cd.plot(10, cd[9], marker='s', markersize=15, color=highlight_color, 
            markerfacecolor=highlight_color, markeredgewidth=1.5, 
            markeredgecolor=color_cd, zorder=4)
# 设置左Y轴（ATE）- 扩大范围让曲线看起来更稳定
ax1.set_ylabel('ATE (cm)', fontsize=30, color=color_ate)
ax1.tick_params(axis='y', labelcolor=color_ate, labelsize=24, pad=5)
ax1.set_ylim([29.00, 29.35])  # 扩大范围
ax1.yaxis.set_major_locator(MultipleLocator(0.05))  # 调整刻度间隔
ax1.yaxis.set_minor_locator(MultipleLocator(0.025))
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 设置右Y轴（CD）- 扩大范围让曲线看起来更稳定
ax1_cd.set_ylabel('CD (cm)', fontsize=30, color=color_cd)
ax1_cd.tick_params(axis='y', labelcolor=color_cd, labelsize=24, pad=5)
ax1_cd.set_ylim([4.8, 5.5])  # 从5.0开始，扩大范围
ax1_cd.yaxis.set_major_locator(MultipleLocator(0.1))  # 调整刻度间隔
ax1_cd.yaxis.set_minor_locator(MultipleLocator(0.05))
ax1_cd.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

# 添加网格
ax1.grid(True, linestyle=':', alpha=0.2, color='gray', which='major')
ax1.grid(True, linestyle=':', alpha=0.2, color='gray', which='minor')

# 添加垂直线标记κ=10（更淡）
ax1.axvline(x=10, color='gray', linestyle='-.', alpha=0.5, linewidth=1.2)

# 添加图例 - 调整位置避免重叠
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc='lower center', fontsize=26, framealpha=0.9, ncols=2,  edgecolor='black', fancybox=False)

# 添加子图标签
ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=32, fontweight='bold', va='top')

# ========== 第二张图：PCG迭代次数 vs κ ==========
ax2.plot(kappa, pcg_iter, marker='^', linestyle='-', color=color_pcg, 
         linewidth=5, markersize=10, markerfacecolor='white', 
         markeredgewidth=1.5, markeredgecolor=color_pcg, zorder=3)

# 高亮κ=10的点
ax2.plot(10, pcg_iter[9], marker='^', markersize=15, color=highlight_color, 
         markerfacecolor=highlight_color, markeredgewidth=1.5, 
         markeredgecolor=color_pcg, zorder=4)

# 设置Y轴
ax2.set_ylabel('Avg. PCG Iterations', fontsize=30)
ax2.tick_params(axis='y', labelsize=24, pad=5)
ax2.set_xlabel('$\\kappa_{tg}$', fontsize=30)  # 简化标签
ax2.set_ylim([8.92, 9.05])
ax2.yaxis.set_major_locator(MultipleLocator(0.02))
ax2.yaxis.set_minor_locator(MultipleLocator(0.01))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

# 设置X轴为对数刻度
ax2.set_xscale('log')
ax2.set_xlim([0.9, 120])
ax2.set_xticks([1, 2, 5, 10, 20, 50, 100])
ax2.get_xaxis().set_major_formatter(FormatStrFormatter('%d'))
ax2.tick_params(axis='both', labelsize=24)

# 添加网格
ax2.grid(True, linestyle=':', alpha=0.2, color='gray', which='major')
ax2.grid(True, linestyle=':', alpha=0.2, color='gray', which='minor', axis='y')

# 添加垂直线标记κ=10（更淡，无文字）
ax2.axvline(x=10, color='gray', linestyle='-.', alpha=0.5, linewidth=1.2)

# 添加子图标签
ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=32, 
         fontweight='bold', va='top')

# 调整布局
plt.tight_layout()
plt.subplots_adjust(hspace=0.15)

# 保存图表
plt.savefig('dcreg_parameter_analysis.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.savefig('dcreg_parameter_analysis.png', format='png', dpi=300, bbox_inches='tight')
plt.show()

# 打印数据统计信息
print("数据统计:")
print(f"ATE - 平均值: {np.mean(ate):.3f} cm, 标准差: {np.std(ate):.3f} cm")
print(f"CD - 平均值: {np.mean(cd):.3f} cm, 标准差: {np.std(cd):.3f} cm")
print(f"PCG Iter - 平均值: {np.mean(pcg_iter):.3f}, 标准差: {np.std(pcg_iter):.3f}")
print(f"\nκ=10时的值:")
print(f"ATE: {ate[9]:.3f} cm")
print(f"CD: {cd[9]:.3f} cm")
print(f"PCG Iterations: {pcg_iter[9]:.3f}")