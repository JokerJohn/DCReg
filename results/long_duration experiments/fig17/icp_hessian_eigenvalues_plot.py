# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib as mpl
# # mpl.rcParams['text.usetex'] = True

# plt.style.use('seaborn-v0_8-whitegrid')
# mpl.rcParams.update({
#     # 'font.size': 22,
#     # 'font.family': 'Arial',
#     # 'axes.labelsize': 24,
#     # 'axes.titlesize': 24,
#     # 'legend.fontsize': 22,
#     # 'xtick.labelsize': 24,
#     # 'ytick.labelsize': 24,
#     'figure.dpi': 300,
#     'savefig.dpi': 300,
#     'savefig.bbox': 'tight',
#     'lines.linewidth': 2.5,
#     'axes.linewidth': 1.5,
#     'axes.grid': True,
#     'grid.alpha': 0.2,
#     'grid.linestyle': '-',
#     'grid.linewidth': 0.8,
#     'axes.edgecolor': 'black',  # 设置坐标轴边框为黑色
#     'axes.labelpad': 10,  # 增加轴标签的padding
#     'legend.edgecolor': 'black',  # 图例边框黑色
#     'legend.framealpha': 1.0,  # 图例不透明
# })

# # Set up the data
# eigenvalues = [923, 12280, 33258, 105, 1446, 289]
# degenerate_mask = [1, 0, 0, 1, 0, 0]
# # eigenvalues = [765, 12425, 26689, 207, 1306, 310]
# # degenerate_mask = [1, 0, 0, 0, 0, 0]
# labels = ['$\mathbf{r}_0$', '$\mathbf{r}_1$', '$\mathbf{r}_2$', '$\mathbf{t}_0$', '$\mathbf{t}_1$', '$\mathbf{t}_2$']

# # Create figure with appropriate size for journal paper
# fig, ax = plt.subplots(figsize=(10, 6))

# # Set font to Arial and size > 20
# plt.rcParams['font.family'] = 'Arial'
# plt.rcParams['font.size'] = 24

# # Create x positions for bars
# x_pos = np.arange(len(labels))

# # Create bars with different colors for degenerate vs non-degenerate
# colors = []
# for i, is_degenerate in enumerate(degenerate_mask):
#     if is_degenerate:
#         colors.append('#FF6B6B')  # Red for degenerate
#     else:
#         colors.append('#4ECDC4')  # Teal for non-degenerate

# bars = ax.bar(x_pos, eigenvalues, color=colors, edgecolor='black', linewidth=1.5)

# # Add hatching pattern to degenerate bars for better visibility in B&W printing
# for i, (bar, is_degenerate) in enumerate(zip(bars, degenerate_mask)):
#     if is_degenerate:
#         bar.set_hatch('///')

# # Set labels and title
# # ax.set_xlabel('Dimension', fontsize=24, fontweight='bold')
# ax.set_ylabel('Eigenvalue', fontsize=28, fontweight='bold')
# # ax.set_title('Hessian Eigenvalues', fontsize=26, fontweight='bold')

# # Set x-axis labels
# ax.set_xticks(x_pos)
# ax.set_xticklabels(labels, fontsize=26)

# # Add vertical line to separate rotation and translation
# separation_x = 2.5  # Between r2 and t0
# ax.axvline(x=separation_x, color='black', linestyle='--', linewidth=2, alpha=0.7)

# # Add text labels for rotation and translation regions
# # ax.text(1, max(eigenvalues) * 0.95, 'Rotation', ha='center', fontsize=22, fontweight='bold')
# # ax.text(4, max(eigenvalues) * 0.95, 'Translation', ha='center', fontsize=22, fontweight='bold')

# # Add legend
# from matplotlib.patches import Patch
# legend_elements = [
#     Patch(facecolor='#FF6B6B', edgecolor='black', hatch='///', label='Deg.'),
#     Patch(facecolor='#4ECDC4', edgecolor='black', label='Non-deg.')
# ]
# ax.legend(handles=legend_elements, loc='upper right', fontsize=26)

# # Add value labels on top of bars
# for i, (bar, value) in enumerate(zip(bars, eigenvalues)):
#     height = bar.get_height()
#     ax.text(bar.get_x() + bar.get_width()/2., height + max(eigenvalues) * 0.01,
#             f'{value}', ha='center', va='bottom', fontsize=26)

# # Set y-axis to logarithmic scale for better visualization of different magnitudes
# ax.set_yscale('log')
# ax.set_ylabel('Eigenvalue (log scale)', fontsize=28, fontweight='bold')

# # Improve grid for readability
# ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
# ax.set_axisbelow(True)

# # Set y-axis limits with some padding
# ax.set_ylim([50, max(eigenvalues) * 2])

# # Adjust tick parameters
# ax.tick_params(axis='both', which='major', labelsize=26)
# ax.tick_params(axis='y', which='minor', labelsize=26)

# # Tight layout to prevent label cutoff
# plt.tight_layout()

# # Save as PDF for journal paper
# # plt.savefig('icp_hessian_eigenvalues_1142.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.savefig('icp_hessian_eigenvalues_1096.pdf', format='pdf', dpi=300, bbox_inches='tight')

# # Also save as PNG for preview
# # plt.savefig('icp_hessian_eigenvalues_1142.png', format='png', dpi=300, bbox_inches='tight')
# plt.savefig('icp_hessian_eigenvalues_1096.png', format='png', dpi=300, bbox_inches='tight')

# # Display the plot
# plt.show()

# print("Plot saved as 'icp_hessian_eigenvalues.pdf' and 'icp_hessian_eigenvalues.png'")
# print("Degenerate dimensions (r0, t0) are marked in red with hatching pattern")


import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# 先设置全局样式
plt.style.use('seaborn-v0_8-whitegrid')

# 在创建图形之前设置所有的rcParams
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 24  # 修正：原来是2224

mpl.rcParams.update({
    'font.size': 24,
    'font.family': 'Arial',
    'axes.labelsize': 28,
    'axes.titlesize': 26,
    'legend.fontsize': 26,
    'xtick.labelsize': 28,  # 增大x轴标签字体（r0-r2, t0-t2）
    'ytick.labelsize': 26,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'axes.edgecolor': 'black',
    'axes.labelpad': 10,
    'legend.edgecolor': 'black',
    'legend.framealpha': 1.0,
})

# Set up the data
# eigenvalues = [923, 12280, 33258, 105, 1446, 289]
# degenerate_mask = [1, 0, 0, 1, 0, 0]
eigenvalues = [765, 12425, 26689, 207, 1306, 310]
degenerate_mask = [1, 0, 0, 0, 0, 0]
labels = ['$\mathbf{r}_0$', '$\mathbf{r}_1$', '$\mathbf{r}_2$', 
          '$\mathbf{t}_0$', '$\mathbf{t}_1$', '$\mathbf{t}_2$']

# 创建图形（在设置rcParams之后）
fig, ax = plt.subplots(figsize=(10, 6))

# Create x positions for bars
x_pos = np.arange(len(labels))

# Create bars with different colors for degenerate vs non-degenerate
colors = []
for i, is_degenerate in enumerate(degenerate_mask):
    if is_degenerate:
        colors.append('#FF6B6B')  # Red for degenerate
    else:
        colors.append('#4ECDC4')  # Teal for non-degenerate

bars = ax.bar(x_pos, eigenvalues, color=colors, edgecolor='black', linewidth=1.5)

# Add hatching pattern to degenerate bars
for i, (bar, is_degenerate) in enumerate(zip(bars, degenerate_mask)):
    if is_degenerate:
        bar.set_hatch('///')

# Set labels
ax.set_ylabel('Eigenvalue (log scale)', fontsize=30, fontweight='bold')

# Set x-axis labels - 增大字体
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=30)  # 增大到30

# Add vertical line to separate rotation and translation
separation_x = 2.5
ax.axvline(x=separation_x, color='black', linestyle='--', linewidth=2, alpha=0.7)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF6B6B', edgecolor='black', hatch='///', label='Deg.'),
    Patch(facecolor='#4ECDC4', edgecolor='black', label='Non-deg.')
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=26)

# Add value labels on top of bars - 增大字体
for i, (bar, value) in enumerate(zip(bars, eigenvalues)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + max(eigenvalues) * 0.01,
            f'{value}', ha='center', va='bottom', fontsize=32, fontweight='bold')  # 增大到32并加粗

# Set y-axis to logarithmic scale
ax.set_yscale('log')

# Improve grid for readability
ax.grid(True, axis='y', alpha=0.2, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Set y-axis limits with some padding
ax.set_ylim([50, max(eigenvalues) * 2])

# Adjust tick parameters - 确保字体大小生效
ax.tick_params(axis='x', which='major', labelsize=30)  # x轴刻度标签大小
ax.tick_params(axis='y', which='major', labelsize=26)  # y轴刻度标签大小
ax.tick_params(axis='y', which='minor', labelsize=22)

# Tight layout to prevent label cutoff
plt.tight_layout()

# Save as PDF for journal paper
# plt.savefig('icp_hessian_eigenvalues_1096.pdf', format='pdf', dpi=300, bbox_inches='tight')
# plt.savefig('icp_hessian_eigenvalues_1096.png', format='png', dpi=300, bbox_inches='tight')

plt.savefig('icp_hessian_eigenvalues_1142.pdf', format='pdf', dpi=300, bbox_inches='tight')
plt.savefig('icp_hessian_eigenvalues_1142.png', format='png', dpi=300, bbox_inches='tight')

# Display the plot
plt.show()

print("Plot saved as 'icp_hessian_eigenvalues_1096.pdf' and 'icp_hessian_eigenvalues_1096.png'")
print("Degenerate dimensions (r0, t0) are marked in red with hatching pattern")
# print("\n字体大小设置：")
# print("- 数值标签: 32pt (加粗)")
# print("- X轴标签 (r0-r2, t0-t2): 30pt")
# print("- Y轴标签: 30pt")
# print("- 图例: 26pt")