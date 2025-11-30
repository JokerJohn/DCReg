#!/usr/bin/env python3
"""
ICP退化检测方法可视化对比脚本
用于对比DCReg、FCN和ME三种方法的性能表现
生成适用于学术论文的高质量图表
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D


# 设置matplotlib参数以生成高质量学术图表
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams.update({
    'font.size': 16,
    'font.family': 'Arial',
    'axes.labelsize': 24,
    'axes.titlesize': 24,
    'legend.fontsize': 22,
    'xtick.labelsize': 24,
    'ytick.labelsize': 24,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.5,
    'axes.grid': True,
    'grid.alpha': 0.2,
    'grid.linestyle': '-',
    'grid.linewidth': 0.8,
    'axes.edgecolor': 'black',  # 设置坐标轴边框为黑色
    'axes.labelpad': 10,  # 增加轴标签的padding
    'legend.edgecolor': 'black',  # 图例边框黑色
    'legend.framealpha': 1.0,  # 图例不透明
})

# 定义专业的配色方案 - 适合顶级期刊的蓝橙色系
# COLOR_DCREG = "#2E75B6"      # 专业蓝色（Nature/Science常用）
# COLOR_DCREG_DIAG = '#E74C3C' # 对比红色（对角条件数）
# COLOR_FCN = "#FF8C42"        # 温暖橙色
# COLOR_ME = '#52C41A'         # 清新绿色
# COLOR_DEGENERATE = '#FFF3E0'  # 浅橙色背景（退化区域）
# COLOR_THRESHOLD = "#1A1A1A"   # 深灰色（阈值线）

# COLOR_DCREG = "#2E75B6"      # Professional blue
# COLOR_DCREG_DIAG = '#E74C3C' # Contrast red (diagonal condition)
# COLOR_FCN = "#FF8C42"        # Warm orange
# COLOR_ME = '#52C41A'         # Fresh green
# COLOR_DEGENERATE = '#FFF3E0'  # Light orange background
# COLOR_THRESHOLD = "#1A1A1A"   # Dark gray (threshold line)

# Main method colors
COLOR_DCREG = "#2563EB"      # Professional blue (Science/Nature style)
COLOR_DCREG_DIAG = '#DC2626' # Contrast red (diagonal condition number)
COLOR_FCN = "#F97316"        # Academic orange
COLOR_FCND = "#F49048"        # Academic orange
COLOR_ME = '#22C55E'         # Academic green
COLOR_DEGENERATE = '#FEF3C7'  # Light amber background (degenerate regions)
COLOR_THRESHOLD = "#374151"   # Dark gray (threshold line)

# 阈值参数
ME_EIGENVALUE_THRESHOLD = 120  # ME方法的特征值阈值
CONDITION_NUMBER_THRESHOLD = 10  # 条件数阈值

# 需要展示组合信息的帧ID
FRAME_IDS_FOR_COMBINATION = [2313, 2923]  # 可以根据需要修改 parkinglot数据集
# FRAME_IDS_FOR_COMBINATION = [1068, 1148]  # 添加更多帧ID以展示组合信息  corridor数据集
# FRAME_IDS_FOR_COMBINATION = [1095, 1141]  # 添加更多帧ID以展示组合信息  corridor数据集
# FRAME_IDS_FOR_COMBINATION = [1284, 2157]  # 添加更多帧ID以展示组合信息 stairs数据集
# FRAME_IDS_FOR_COMBINATION = [3373, 6597]  # 添加更多帧ID以展示组合信息 cave02数据集

# 多数据集路径配置（用户需要根据实际情况修改）
DATASET_CONFIGS = [
    {
        'name': 'Parking Lot',
        'dcreg': '/home/xchu/data/ltloc_result/detection_results/parkinglot_raw_dcreg_none/data_icp.txt',
        'fcn': '/home/xchu/data/ltloc_result/detection_results/parkinglot_raw_fcn_none/data_icp.txt',
        'me': '/home/xchu/data/ltloc_result/detection_results/parkinglot_raw_me_sr_none/data_icp.txt'
    },
    {
        'name': 'Corridor',
        'dcreg': '/home/xchu/data/ltloc_result/detection_results/20220216_corridor_day_ref_ours/data_icp.txt',
        'fcn': '/home/xchu/data/ltloc_result/detection_results/20220216_corridor_day_ref_fcn_sr/data_icp.txt',
        'me': '/home/xchu/data/ltloc_result/detection_results/20220216_corridor_day_ref_me_none/data_icp.txt'
    },
    { 
        'name': 'Stairs',
        'dcreg': '/home/xchu/data/ltloc_result/detection_results/stairs_bob_pcg_none/data_icp.txt',
        'fcn': '/home/xchu/data/ltloc_result/detection_results/stairs_bob_fcn_none/data_icp.txt',
        'me': '/home/xchu/data/ltloc_result/detection_results/stairs_bob_me_sr_none/data_icp.txt'
    },
    { 
        'name': 'Cave02',
        'dcreg': '/home/xchu/data/ltloc_result/detection_results/lauren_cavern02_ours_none/data_icp.txt',
        'fcn': '/home/xchu/data/ltloc_result/detection_results/lauren_cavern02_fcn_none/data_icp.txt',
        'me': '/home/xchu/data/ltloc_result/detection_results/lauren_cavern02_me_none/data_icp.txt'
    }
]

dcreg_file = DATASET_CONFIGS[0]['dcreg']
fcn_file = DATASET_CONFIGS[0]['fcn']    
me_file = DATASET_CONFIGS[0]['me']

def load_icp_data(filepath):
    """
    加载ICP数据文件
    
    参数:
        filepath: 数据文件路径
    
    返回:
        DataFrame: 包含所有数据的DataFrame
    """
    # 定义列名
    columns = [
        'timestamp',                    # 0
        'cond_schur_rot',              # 1
        'cond_schur_trans',            # 2
        'cond_diag_rot',               # 3
        'cond_diag_trans',             # 4
        'icp_iterations',              # 5
        'rmse',                        # 6
        'fitness',                     # 7
        'time_cost',                   # 8
        'is_degenerate',               # 9
        'deg_rot1', 'deg_rot2', 'deg_rot3',      # 10-12
        'deg_trans1', 'deg_trans2', 'deg_trans3', # 13-15
        'eigen1', 'eigen2', 'eigen3',            # 16-18
        'eigen4', 'eigen5', 'eigen6'             # 19-21
    ]
    
    try:
        # 尝试读取扩展的DCReg数据（包含方向组合信息）
        data = pd.read_csv(filepath, sep='\s+', header=None)
        
        # 基本列
        base_columns = columns.copy()
        
        # 如果有额外的列（DCReg的方向组合信息）
        if data.shape[1] > 22:
            # 为额外的列添加名称
            for i in range(22, data.shape[1]):
                base_columns.append(f'extra_col_{i}')
        
        data.columns = base_columns[:data.shape[1]]
        # 添加帧索引
        data['frame_index'] = np.arange(len(data))
        return data
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None

def plot_dcreg_condition_number_comparison(dcreg_data, output_path='condition_number_comparison.pdf'):
    """
    绘制DCReg的舒尔补条件数与对角条件数对比图，标注退化区域
    """
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 1, figure=fig, hspace=0.1)
    
    frames = dcreg_data['frame_index']
    
    # 子图A：旋转条件数
    ax1 = fig.add_subplot(gs[0, 0])
    
    # 标注退化区域（基于舒尔补条件数）
    degenerate_regions = dcreg_data['cond_schur_rot'] > CONDITION_NUMBER_THRESHOLD
    for i in range(len(frames)):
        if degenerate_regions.iloc[i]:
            ax1.axvspan(frames.iloc[i]-0.5, frames.iloc[i]+0.5, 
                       alpha=0.2, color=COLOR_DEGENERATE, zorder=1)
    
    # 绘制舒尔补条件数曲线
    ax1.semilogy(frames, dcreg_data['cond_schur_rot'], 
                 label='Schur', color=COLOR_DCREG, linewidth=3, zorder=3)
    
    # 绘制对角条件数曲线
    ax1.semilogy(frames, dcreg_data['cond_diag_rot'], 
                 label='Diagonal', color=COLOR_DCREG_DIAG, linewidth=2.5, 
                 linestyle='--', alpha=0.8, zorder=2)
    
    # 添加阈值线
    ax1.axhline(y=CONDITION_NUMBER_THRESHOLD, color=COLOR_THRESHOLD, linestyle=':', 
                linewidth=3, label=f'Threshold ({CONDITION_NUMBER_THRESHOLD})', zorder=4)
    
    # i do not want to show this axhline in legend
    # handles, labels = ax1.get_legend_handles_labels()
    # handles = [h for h, l in zip(handles, labels) if l != f'Treshold ({CONDITION_NUMBER_THRESHOLD})']
    # labels = [l for l in labels if l != f'Threshold ({CONDITION_NUMBER_THRESHOLD})']
    # ax1.legend(handles, labels, loc='upper left', frameon=True, fancybox=False, shadow=False, 
    #            edgecolor='black', borderpad=0.8)
    
    ax1.set_ylabel('Rot. Cond.', fontsize=26, fontfamily='Arial')
    # ax1.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, ncol=3,
    #            edgecolor='black', borderpad=0.8)
    ax1.set_xlim(frames.min(), frames.max())
    ax1.set_ylim(1, max(dcreg_data['cond_schur_rot'].max(), dcreg_data['cond_diag_rot'].max()) * 3)
    
    # 移除上方子图的x轴标签
    ax1.set_xticklabels([])
    ax1.set_xlabel('')
    
    # 设置边框颜色
    for spine in ax1.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.25)
    
    # 子图B：平移条件数
    ax2 = fig.add_subplot(gs[1, 0])
    
    # 标注退化区域（基于舒尔补条件数）
    degenerate_regions_trans = dcreg_data['cond_schur_trans'] > CONDITION_NUMBER_THRESHOLD
    for i in range(len(frames)):
        if degenerate_regions_trans.iloc[i]:
            ax2.axvspan(frames.iloc[i]-0.5, frames.iloc[i]+0.5, 
                       alpha=0.2, color=COLOR_DEGENERATE, zorder=1)
    
    # 绘制舒尔补条件数曲线
    ax2.semilogy(frames, dcreg_data['cond_schur_trans'], 
                 label='Schur', color=COLOR_DCREG, linewidth=3, zorder=3)
    
    # 绘制对角条件数曲线
    ax2.semilogy(frames, dcreg_data['cond_diag_trans'], 
                 label='Diagonal', color=COLOR_DCREG_DIAG, linewidth=2.5, 
                 linestyle='--', alpha=0.8, zorder=2)
    
    # 添加阈值线
    ax2.axhline(y=CONDITION_NUMBER_THRESHOLD, color=COLOR_THRESHOLD, linestyle=':', 
                linewidth=2.5, label=f'Threshold ({CONDITION_NUMBER_THRESHOLD})', zorder=4)
    
    ax2.set_xlabel('Frame ID', fontsize=26, labelpad=15)
    ax2.set_ylabel('Trans. Cond.', fontsize=26, fontfamily='Arial')
    ax2.tick_params(axis='x', which='major', pad=10)
    # ax2.set_xticklabels(frames, fontsize=26, fontfamily='Arial')

    # i do not want to show this ax2.axhline in legend
    handles, labels = ax2.get_legend_handles_labels()
    handles = [h for h, l in zip(handles, labels) if l != f'Threshold ({CONDITION_NUMBER_THRESHOLD})']
    labels = [l for l in labels if l != f'Threshold ({CONDITION_NUMBER_THRESHOLD})']
    ax2.legend(handles, labels, loc='upper left', frameon=True, fancybox=False, shadow=False, ncol=2,
               edgecolor='black', borderpad=0.8)

    # ax2.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, 
    #            edgecolor='black', borderpad=0.8)
    # ax2.set_xlim(frames.min(), frames.max())
    ax2.set_ylim(1, max(dcreg_data['cond_schur_trans'].max(), dcreg_data['cond_diag_trans'].max()) * 3)
    
    # 设置边框颜色
    for spine in ax2.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.25)
    
    # 保存图表
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved condition number comparison to {output_path}")

# def plot_degeneracy_timeline_improved(dcreg_data, fcn_data, me_data, output_path='degeneracy_detection_timeline.pdf'):
#     """
#     绘制改进的退化检测结果时序图
#     DCReg使用t0-t2, r0-r2标记（t在上，r在下），FCN和ME使用v₀-v₅标记
#     """
#     fig = plt.figure(figsize=(16, 18))
#     gs = GridSpec(3, 1, figure=fig, hspace=0.21)
    
#     # 准备数据
#     deg_cols = ['deg_rot1', 'deg_rot2', 'deg_rot3', 'deg_trans1', 'deg_trans2', 'deg_trans3']
    
#     # 处理ME数据：每个特征值与阈值比较
#     me_deg_matrix = np.zeros((6, len(me_data)))
#     eigen_cols = ['eigen1', 'eigen2', 'eigen3', 'eigen4', 'eigen5', 'eigen6']
#     for i, eigen_col in enumerate(eigen_cols):
#         me_deg_matrix[i, :] = (me_data[eigen_col] < ME_EIGENVALUE_THRESHOLD).astype(int)
    
#     # the dcreg_reordered_cols for fcn-sr should be
    
#     # 重新排序DCReg数据：t0-t2在上，r0-r2在下
#     dcreg_reordered_cols = ['deg_trans1', 'deg_trans2', 'deg_trans3', 'deg_rot1', 'deg_rot2', 'deg_rot3']
    
#     # 定义标签（使用下标）
#     dcreg_labels = ['$\mathbf{t_0}$', '$\mathbf{t_1}$', '$\mathbf{t_2}$', '$\mathbf{r_0}$', '$\mathbf{r_1}$', '$\mathbf{r_2}$']  # t在上，r在下
#     fcn_me_labels = ['$\mathbf{v_0}$', '$\mathbf{v_1}$', '$\mathbf{v_2}$', '$\mathbf{v_3}$', '$\mathbf{v_4}$', '$\mathbf{v_5}$']  # 使用下标

#     datasets = [
#         (fcn_data[deg_cols].values.T, 'FCN', COLOR_FCN, fcn_me_labels),
#         (me_deg_matrix, 'ME', COLOR_ME, fcn_me_labels),
#         (dcreg_data[dcreg_reordered_cols].values.T, 'Ours', COLOR_DCREG, dcreg_labels)
#     ]
    
#     # 自定义颜色映射
#     from matplotlib.colors import ListedColormap
    
#     # 为每种方法创建子图
#     for idx, (deg_matrix, method_name, method_color, labels) in enumerate(datasets):
#         ax = fig.add_subplot(gs[idx, 0])
        
#         # 创建自定义颜色映射（白色和方法对应的颜色）
#         colors = ['white', method_color]
#         cmap = ListedColormap(colors)
        
#         # 创建热图
#         im = ax.imshow(deg_matrix, aspect='auto', cmap=cmap, 
#                       interpolation='nearest', vmin=0, vmax=1)
        
#         # 设置y轴
#         ax.set_title(f'({chr(97+idx)}) {method_name}', 
#                     loc='center', fontsize=32, fontweight='bold', pad=15)
        
#         # 设置边框颜色
#         for spine in ax.spines.values():
#             spine.set_edgecolor('black')
#             spine.set_linewidth(1.25)
        
#         # 设置y轴刻度
#         ax.set_yticks(range(6))
#         ax.set_yticklabels(labels, fontsize=30, fontfamily='Arial')
#         ax.tick_params(axis='y', which='major', pad=10)
        
#         # 只在最后一个子图显示x轴标签
#         if idx == 2:
#             ax.set_xlabel('Frame ID', fontsize=30, labelpad=15)
#             # 设置x轴刻度
#             frame_count = deg_matrix.shape[1]
#             x_ticks = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
#             ax.set_xticks(x_ticks)
#             ax.set_xticklabels(x_ticks, fontsize=28, fontfamily='Arial')
#             ax.tick_params(axis='x', which='major', pad=15)

#         else:
#             ax.set_xticklabels([])
        
#         # 添加网格线
#         for x in range(0, deg_matrix.shape[1], 20):
#             ax.axvline(x=x-0.5, color='gray', alpha=0.15, linewidth=0.5)
#         for y in range(7):
#             ax.axhline(y=y-0.5, color='gray', alpha=0.15, linewidth=0.5)
        
#         # 添加统计信息（简化版）
#         total_deg = np.sum(deg_matrix)
#         deg_percentage = (total_deg / deg_matrix.size) * 100
        
#         if idx == 2:  # DCReg
#             # 重新计算旋转和平移的统计（注意顺序已改变）
#             per_direction_rates = (np.sum(deg_matrix, axis=1) / deg_matrix.shape[1]) * 100
#             trans_rate = np.mean(per_direction_rates[:3])  # 前三个是平移
#             rot_rate = np.mean(per_direction_rates[3:])    # 后三个是旋转
#             stats_text = f'{deg_percentage:.1f}% (T:{trans_rate:.1f}%, R:{rot_rate:.1f}%)'
#         else:
#             stats_text = f'{deg_percentage:.1f}%'
        
#         ax.text(0.03, 0.20, stats_text,
#                transform=ax.transAxes, ha='left', va='top',
#                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
#                         edgecolor='black', alpha=0.9, linewidth=1.5),
#                fontsize=30)
    
#     # 保存图表
#     plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
#     plt.close()
#     print(f"Saved improved degeneracy timeline to {output_path}")

def plot_degeneracy_timeline_improved(dcreg_data, fcn_data, me_data, output_path='degeneracy_detection_timeline.pdf'):
    """
    绘制改进的退化检测结果时序图
    DCReg使用t0-t2, r0-r2标记（t在上，r在下），FCN和ME使用v0-v5标记
    """
    fig = plt.figure(figsize=(16, 18))
    gs = GridSpec(3, 1, figure=fig, hspace=0.18)
    
    # 准备数据
    deg_cols = ['deg_rot1', 'deg_rot2', 'deg_rot3', 'deg_trans1', 'deg_trans2', 'deg_trans3']
    
    # 处理ME数据：每个特征值与阈值比较
    me_deg_matrix = np.zeros((6, len(me_data)))
    eigen_cols = ['eigen1', 'eigen2', 'eigen3', 'eigen4', 'eigen5', 'eigen6']
    for i, eigen_col in enumerate(eigen_cols):
        me_deg_matrix[i, :] = (me_data[eigen_col] < ME_EIGENVALUE_THRESHOLD).astype(int)

    # 处理fcn数据：每个特征值与COND阈值比较
    fcn_deg_matrix = np.zeros((6, len(fcn_data)))
    for i, col in enumerate(eigen_cols):
        fcn_deg_matrix[i, :] = (fcn_data['eigen6'] / fcn_data[col] > CONDITION_NUMBER_THRESHOLD).astype(int)


    # 重新排序DCReg数据：t0-t2在上，r0-r2在下
    dcreg_reordered_cols = ['deg_trans1', 'deg_trans2', 'deg_trans3', 'deg_rot1', 'deg_rot2', 'deg_rot3']
    
    # 定义标签（使用LaTeX下标）
    dcreg_labels = [r'$\mathbf{t_0}$', r'$\mathbf{t_1}$', r'$\mathbf{t_2}$', r'$\mathbf{r_0}$', r'$\mathbf{r_1}$', r'$\mathbf{r_2}$']
    fcn_me_labels = [r'$\mathbf{v_0}$', r'$\mathbf{v_1}$', r'$\mathbf{v_2}$', r'$\mathbf{v_3}$', r'$\mathbf{v_4}$', r'$\mathbf{v_5}$']


    datasets = [
        # (fcn_data[deg_cols].values.T, 'FCN', COLOR_FCN, fcn_me_labels),
        (fcn_deg_matrix, 'FCN', COLOR_FCND, fcn_me_labels),
        # (me_deg_matrix, 'ME', COLOR_ME, fcn_me_labels),
        # (dcreg_data[dcreg_reordered_cols].values.T, 'Ours', COLOR_DCREG, dcreg_labels)
        (me_deg_matrix, 'ME', COLOR_FCND, fcn_me_labels),
        (dcreg_data[dcreg_reordered_cols].values.T, 'Ours', COLOR_FCND, dcreg_labels)
    ]
    
    # 自定义颜色映射
    from matplotlib.colors import ListedColormap
    
    # 为每种方法创建子图
    for idx, (deg_matrix, method_name, method_color, labels) in enumerate(datasets):
        ax = fig.add_subplot(gs[idx, 0])
        
        # 创建自定义颜色映射（白色和方法对应的颜色）
        colors = ['white', method_color]
        cmap = ListedColormap(colors)
        
        # 创建热图
        im = ax.imshow(deg_matrix, aspect='auto', cmap=cmap, 
                      interpolation='nearest', vmin=0, vmax=1)
        
        # 设置y轴
        ax.set_title(f'({chr(97+idx)}) {method_name}', 
                    loc='center', fontsize=32, fontweight='bold', pad=15)
        
        # 设置边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.25)
        
        # 设置y轴刻度
        ax.set_yticks(range(6))
        ax.set_yticklabels(labels, fontsize=30, fontfamily='Arial')
        ax.tick_params(axis='y', which='major', pad=10)
        
        # 只在最后一个子图显示x轴标签
        if idx == 2:
            ax.set_xlabel('Frame ID', fontsize=30, labelpad=15)
            # 设置x轴刻度
            frame_count = deg_matrix.shape[1]
            x_ticks = np.linspace(0, frame_count-1, min(10, frame_count), dtype=int)
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_ticks, fontsize=28, fontfamily='Arial')
            ax.tick_params(axis='x', which='major', pad=15)
        else:
            ax.set_xticklabels([])
        
        # 添加网格线
        for x in range(0, deg_matrix.shape[1], 20):
            ax.axvline(x=x-0.5, color='gray', alpha=0.15, linewidth=0.5)
        for y in range(7):
            ax.axhline(y=y-0.5, color='gray', alpha=0.15, linewidth=0.5)
        
        # 添加统计信息（简化版）
        total_deg = np.sum(deg_matrix)
        deg_percentage = (total_deg / deg_matrix.size) * 100
        
        # if idx == 2:  # DCReg
        #     # 重新计算旋转和平移的统计（注意顺序已改变）
        #     per_direction_rates = (np.sum(deg_matrix, axis=1) / deg_matrix.shape[1]) * 100
        #     trans_rate = np.mean(per_direction_rates[:3])  # 前三个是平移
        #     rot_rate = np.mean(per_direction_rates[3:])    # 后三个是旋转
        #     stats_text = f'{deg_percentage:.1f}% (T:{trans_rate:.1f}%, R:{rot_rate:.1f}%)'
        # else:
        #     stats_text = f'{deg_percentage:.1f}%'
        
        # ax.text(0.03, 0.20, stats_text,
        #        transform=ax.transAxes, ha='left', va='top',
        #        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
        #                 edgecolor='black', alpha=0.9, linewidth=1.5),
        #        fontsize=30)
    
    # 保存图表
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved improved degeneracy timeline to {output_path}")

def plot_degeneracy_ratio_comparison_single(dcreg_data, fcn_data, me_data, output_path='degeneracy_ratio_comparison.pdf'):
    """
    绘制单个数据集的退化比例对比图
    改进样式，调整v5空间分配
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 准备数据
    deg_cols_dcreg = ['deg_rot1', 'deg_rot2', 'deg_rot3', 'deg_trans1', 'deg_trans2', 'deg_trans3']
    
    # 计算DCReg的退化比例
    dcreg_ratios = (dcreg_data[deg_cols_dcreg].sum() / len(dcreg_data)) * 100
    
    # 计算FCN的退化比例
    # fcn_ratios = (fcn_data[deg_cols_dcreg].sum() / len(fcn_data)) * 100
    
    # 计算ME的退化比例（基于特征值阈值）
    me_ratios = []
    eigen_cols = ['eigen1', 'eigen2', 'eigen3', 'eigen4', 'eigen5', 'eigen6']
    for eigen_col in eigen_cols:
        ratio = (np.sum(me_data[eigen_col] < ME_EIGENVALUE_THRESHOLD) / len(me_data)) * 100
        me_ratios.append(ratio)
    me_ratios = np.array(me_ratios)

    # 计算fcn的退化比例（基于cond阈值）
    fcn_ratios = []
    for eigen_col in eigen_cols:
        ratio = (np.sum(fcn_data['eigen6'] / fcn_data[eigen_col] > CONDITION_NUMBER_THRESHOLD) / len(fcn_data)) * 100
        fcn_ratios.append(ratio)
    fcn_ratios = np.array(fcn_ratios)   

    
    # 调整x轴位置，如果v5为空，给前面的留更多空间
    if dcreg_ratios[5] < 0.1 and fcn_ratios[5] < 0.1 and me_ratios[5] < 0.1:
        # v5为空，调整间距
        x_positions = np.array([0, 1.2, 2.4, 3.6, 4.8, 6.0])
    else:
        x_positions = np.arange(6)
    
    width = 0.4
    
    # 绘制条形图
    bars1 = ax.bar(x_positions - width, dcreg_ratios, width, label='Ours', 
                    color=COLOR_DCREG, alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x_positions, fcn_ratios, width, label='FCN', 
                    color=COLOR_FCN, alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x_positions + width, me_ratios, width, label='ME', 
                    color=COLOR_ME, alpha=0.9, edgecolor='white', linewidth=1)
    
    # 添加数值标签（改进样式）
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.1:  # 显示所有大于0.1%的值
                # 格式化显示
                if height < 1:
                    text = f'{height:.1f}'
                else:
                    text = f'{height:.1f}'
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       text, ha='center', va='bottom', fontsize=18, fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # 设置轴标签
    ax.set_ylabel('Degeneracy Rate (%)', fontsize=26)
    ax.set_yticks([])
    
    # 设置x轴刻度标签（使用LaTeX下标）
    ax.set_xticks(x_positions)
    x_labels = [r'$\mathbf{v_0}$', r'$\mathbf{v_1}$', r'$\mathbf{v_2}$', r'$\mathbf{v_3}$', r'$\mathbf{v_4}$', r'$\mathbf{v_5}$']
    ax.set_xticklabels(x_labels, fontsize=30)
    ax.tick_params(axis='x', which='major', pad=15)
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, fancybox=False, 
              shadow=False, fontsize=22, ncol=1, edgecolor='black', borderpad=0.8)
    ax.set_ylim(0, max(max(dcreg_ratios), max(fcn_ratios), max(me_ratios)) * 1.1)
    
    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.set_axisbelow(True)
    
    # 设置边框颜色
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # 保存图表
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved degeneracy ratio comparison to {output_path}")





def plot_eigen_direction_combination_origin(dcreg_data, frame_ids=None, output_path='eigen_direction_combination.pdf'):
    """
    绘制特定帧的特征方向组合信息
    展示每个特征方向是如何由基本运动方向组合而成的
    数据格式：从第22列开始，每6列代表一个特征方向的信息
    """
    if frame_ids is None:
        frame_ids = FRAME_IDS_FOR_COMBINATION
    
    # 确保只选择两帧
    if len(frame_ids) > 2:
        frame_ids = frame_ids[:2]
    elif len(frame_ids) < 2:
        print("Need at least 2 frame IDs for comparison")
        return
    
    # 检查数据列数
    if dcreg_data.shape[1] < 60:
        print(f"Warning: DCReg data has only {dcreg_data.shape[1]} columns, expected at least 60")
        return
    
    # 创建图表 - 减小子图间距
    fig, axes = plt.subplots(2, 1, figsize=(9, 14))
    fig.subplots_adjust(hspace=0.15)  # 减小子图间距

    # 定义专业的配色方案 - 橙蓝浅色系
    motion_colors = {
        0: '#FF8C42',   # Roll - 温暖橙色
        1: '#2E75B6',   # Pitch - 专业蓝色
        2: '#FFB366',   # Yaw - 浅橙色
        3: '#5B9BD5',   # X - 浅蓝色
        4: '#FDD0A2',   # Y - 非常浅的橙色
        5: '#9DC3E6'    # Z - 非常浅的蓝色
    }

        # Motion direction colors - distinct families for rotation vs translation
    # motion_colors = {
    #     # Rotation components (blue family - matching DCReg)
    #     0: '#1E5A8E',   # Roll - Deep blue
    #     1: '#2E75B6',   # Pitch - Professional blue (same as DCReg)
    #     2: '#7EB4E2',   # Yaw - Light blue
        
    #     # Translation components (orange/green family - matching FCN/ME)
    #     3: '#FF8C42',   # X - Warm orange (same as FCN)
    #     4: '#52C41A',   # Y - Fresh green (same as ME)
    #     5: '#8B6F47',   # Z - Earth brown (complementary)
    # }
    # motion_colors = {
    #     # Rotation components (blue/purple family)
    #     0: '#3B82F6',   # Roll - Medium blue
    #     1: '#6366F1',   # Pitch - Indigo blue
    #     2: '#8B5CF6',   # Yaw - Purple
        
    #     # Translation components (orange/green family)  
    #     3: '#F59E0B',   # X - Amber
    #     4: '#10B981',   # Y - Emerald green
    #     5: '#06B6D4'    # Z - Cyan
    # }
    
    motion_labels = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']
    
    # 用于存储图例句柄
    legend_handles = []
    
    for idx, (frame_id, ax) in enumerate(zip(frame_ids, axes)):
        if frame_id >= len(dcreg_data):
            print(f"Frame {frame_id} out of range")
            continue
            
        frame_row = dcreg_data.iloc[frame_id]
        
        # 准备数据：6个特征方向（3个旋转，3个平移）
        eigen_labels = []
        motion_compositions = []
        angles = []
        is_degenerate = []
        
        # 解析每个特征方向的数据
        for i in range(6):
            base_col = 22 + i * 6
            
            # 2 279693.298891 41.566539564 50.6402922279 44.6612314274 4.69847634469 
            # 1 260788.297239 41.3504508696 46.2311346842 52.5487291007 1.22013621506 
            # 0 1491.82786322 4.10431415538 5.81540741581 2.9990078145 91.1855847697 
            # 0 2.74365414348 3.89043578552 92.1000442241 1.94693650535 5.95301927059 
            # 1 26.126015931 2.94526604605 1.69052206648 93.7898128445 4.51966508907
            # 2 3291.03175228 4.61532956816 5.89861371867 4.22477082053 89.8766154608
            if base_col + 5 < len(frame_row):
                # 读取运动方向ID
                motion_id = int(frame_row.iloc[base_col])
                eigenvalue= frame_row.iloc[base_col + 1]


                # 读取与主运动方向的夹角
                angle = frame_row.iloc[base_col + 2]

                print(f"Processing frame {frame_id}, motion ID: {motion_id}, eigenvalue: {eigenvalue}, angle: {angle}, frame_row_size: {len(frame_row)}")
                # print(f"Motion ID: {motion_id}, Angle: {angle}, Base Column: {base_col}")
                # print the frame line between 22 and 60
                print(frame_row.iloc[22:60])

                # 读取运动组合比例（3个分量）
                roll_pct = frame_row.iloc[base_col + 3] if i < 3 else 0
                pitch_pct = frame_row.iloc[base_col + 4] if i < 3 else 0
                yaw_pct = frame_row.iloc[base_col + 5] if i < 3 else 0
                
                if i >= 3:  # 平移方向
                    x_pct = frame_row.iloc[base_col + 3]
                    y_pct = frame_row.iloc[base_col + 4]
                    z_pct = frame_row.iloc[base_col + 5]
                    motion_comp = [0, 0, 0, x_pct, y_pct, z_pct]
                else:  # 旋转方向
                    motion_comp = [roll_pct, pitch_pct, yaw_pct, 0, 0, 0]
                
                # 标签
                if i < 3:
                    eigen_labels.append(r'$\mathbf{r_' + str(i) + '}$')
                    is_deg = frame_row[f'deg_rot{i+1}'] > 0
                else:
                    eigen_labels.append(r'$\mathbf{t_' + str(i-3) + '}$')
                    is_deg = frame_row[f'deg_trans{i-3+1}'] > 0
                
                motion_compositions.append(motion_comp)
                angles.append(angle)
                is_degenerate.append(is_deg)
                print(f"Frame {frame_id}, Motion {i}: Angle = {angle}, Is Degenerate = {is_deg}")
            else:
                print(f"Warning: Frame {frame_id} does not have enough data for motion {i}")
            #     # 如果数据不足，使用默认值
            #     eigen_labels.append(r'$\mathbf{v_' + str(i) + '}$')
            #     motion_compositions.append([0] * 6)
            #     angles.append(0)
            #     is_degenerate.append(False)
        
        # 转换为numpy数组
        motion_compositions = np.array(motion_compositions).T  # 6x6 -> 6x6 转置
        
        # 创建堆叠条形图
        x = np.arange(len(eigen_labels))
        width = 0.65
        
        # 绘制堆叠条形图
        bottom = np.zeros(len(eigen_labels))
        for j, (comp_row, label) in enumerate(zip(motion_compositions, motion_labels)):
            bars = ax.bar(x, comp_row, width, bottom=bottom, 
                          label=label if idx == 0 else None,  # 只在第一个子图创建图例
                          color=motion_colors[j], 
                          alpha=0.95, edgecolor='white', linewidth=1.5)
            
            # 收集图例句柄（只从第一个子图）
            if idx == 0 and j < 6:
                legend_handles.append(bars[0])
            
            # 添加百分比标签（只显示>10%的）
            for k, (height, btm) in enumerate(zip(comp_row, bottom)):
                if height > 10:
                    ax.text(k, btm + height/2, f'{height:.1f}', 
                           ha='center', va='center', fontsize=24, fontweight='bold')
            
            bottom += comp_row
        
        # 在每个条形顶部添加角度信息和退化标记
        for j, (x_pos, angle, is_deg) in enumerate(zip(x, angles, is_degenerate)):
            # 使用更专业的标记：三角形表示退化，圆形表示非退化
            if is_deg:
                marker = '●'
                color = '#E74C3C'  # 红色
            else:
                marker = '○'
                color = '#1A1A1A'  # 深灰色

            ax.text(x_pos, 102, f'{angle:.1f}$\degree$', 
                   ha='center', va='bottom', fontsize=24)
            ax.text(x_pos, 108, marker, 
                   ha='center', va='bottom', fontsize=24, color=color)
        
        # 设置轴属性
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(eigen_labels, fontsize=28, fontfamily='Arial')
        ax.tick_params(axis='x', which='major', pad=10)
        ax.tick_params(axis='y', which='major', pad=5)
        
        # 只在左侧子图显示y轴标签
        if idx == 0:
            # ax.set_ylabel('Composition (%)', fontsize=26)
            ax.set_xticklabels([])
            ax.set_ylabel(f'Frame {frame_id+1}', fontsize=30, fontweight='bold', labelpad=0)
        else:
            ax.set_ylabel('')
            # ax.set_yticklabels([])
            ax.set_ylabel(f'Frame {frame_id+1}', fontsize=30, fontweight='bold', labelpad=0)

        
        # 设置x轴标题（Frame信息）
        
        # 添加网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)
        
        # 设置边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    
    # 添加统一的图例到两个子图的正中心上方
    # 创建退化标记的图例项
    # from matplotlib.lines import Line2D
    # degenerate_marker = Line2D([0], [0], marker='●', color='w', 
    #                           markerfacecolor='#E74C3C', markersize=12, 
    #                           label='Degenerate', linestyle='None')
    # non_degenerate_marker = Line2D([0], [0], marker='○', color='w', 
    #                               markerfacecolor='#1A1A1A', markersize=12, 
    #                               label='Non-degenerate', linestyle='None')
    
    # 合并所有图例项
    # all_handles = legend_handles + [degenerate_marker, non_degenerate_marker]
        all_handles = legend_handles
    all_labels = motion_labels + ['Degenerate', 'Non-degenerate']
    
    # 在图的顶部中心创建图例
    fig.legend(handles=all_handles, labels=all_labels, 
              loc='upper center', bbox_to_anchor=(0.56, 1.0),
              ncol=3, fontsize=24, frameon=True, 
              edgecolor='black', borderpad=0.8)
    
    # 调整布局以给图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.88)
    
    # 保存图表
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved eigen direction combination to {output_path}")

def plot_eigen_direction_combination(dcreg_data, frame_ids=None, output_path='eigen_direction_combination.pdf'):
    """
    绘制特定帧的特征方向组合信息
    展示每个特征方向是如何由基本运动方向组合而成的
    数据格式：从第22列开始，每6列代表一个特征方向的信息
    """
    if frame_ids is None:
        frame_ids = FRAME_IDS_FOR_COMBINATION
    
    # 确保只选择两帧
    if len(frame_ids) > 2:
        frame_ids = frame_ids[:2]
    elif len(frame_ids) < 2:
        print("Need at least 2 frame IDs for comparison")
        return
    
    # 检查数据列数
    if dcreg_data.shape[1] < 60:
        print(f"Warning: DCReg data has only {dcreg_data.shape[1]} columns, expected at least 60")
        return
    
    # 创建图表 - 减小子图间距
    fig, axes = plt.subplots(2, 1, figsize=(9, 14))
    fig.subplots_adjust(hspace=0.15)  # 减小子图间距

    # 定义专业的配色方案 - 区分旋转和平移
    # motion_colors = {
    #     # 旋转组分 (蓝色系)
    #     # 0: '#5B9BD5',   # Roll - 中蓝色
    #     # 1: '#2E75B6',   # Pitch - 深蓝色
    #     # 2: '#A5C8E4',   # Yaw - 浅蓝色
    #     0: '#E94849',   # Roll - 深蓝色
    #     1: '#5889B0',   # Pitch - 专业蓝色
    #     2: '#7EBF6E',   # Yaw - 浅蓝色
        
    #     # 平移组分 (橙绿色系)
    #     3: '#F4B183',   # X - 浅橙色
    #     4: '#70AD47',   # Y - 绿色
    #     5: '#FFC000'    # Z - 金黄色
    # }
    # motion_colors = {
    # # 旋转组分 (方案一)
    #     0: '#F39C12',   # Roll - 琥珀橙
    #     1: '#9B59B6',   # Pitch - 紫罗兰
    #     2: '#1ABC9C',   # Yaw - 青绿色
        
    #     # 平移组分 (方案一)
    #     3: '#E74C3C',   # X - 朱红色
    #     4: '#3498DB',   # Y - 天蓝色
    #     5: '#2ECC71'    # Z - 翠绿色
    # }
    

# 方案一（推荐）
    # motion_colors = {
    #     # 旋转组分 (冷色调渐变)
    #     0: '#87CEEB',   # Roll - 天蓝色
    #     1: '#6495ED',   # Pitch - 矢车菊蓝
    #     2: '#9370DB',   # Yaw - 中紫色
        
    #     # 平移组分 (暖色调渐变)
    #     3: '#FF9999',   # X - 珊瑚红
    #     4: '#FFB366',   # Y - 杏橙色
    #     5: '#FFD700'    # Z - 金黄色
    # }


    motion_colors = {
        0: '#FF8C42',   # Roll - 温暖橙色
        1: '#5B9BD5',   # Y - 浅蓝色
        2: '#FDD0A2',   # Y - 非常浅的橙色

        3: '#2E75B6',   # pitch - 专业蓝色
        4: '#FFB366',   # X - 浅橙色
        5: '#9DC3E6'    # Z - 非常浅的蓝色
    }

    motion_labels = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']
    
    # 用于存储图例句柄
    legend_handles = []
    
    for idx, (frame_id, ax) in enumerate(zip(frame_ids, axes)):
        if frame_id >= len(dcreg_data):
            print(f"Frame {frame_id} out of range")
            continue
            
        frame_row = dcreg_data.iloc[frame_id]
        
        # 准备数据：6个特征方向（3个旋转，3个平移）
        eigen_labels = []
        motion_compositions = []
        angles = []
        is_degenerate = []
        
        # 解析每个特征方向的数据
        for i in range(6):
            base_col = 22 + i * 6
            
            if base_col + 5 < len(frame_row):
                # 读取运动方向ID
                motion_id = int(frame_row.iloc[base_col])
                eigenvalue= frame_row.iloc[base_col + 1]

                # 读取与主运动方向的夹角
                angle = frame_row.iloc[base_col + 2]

                print(f"Processing frame {frame_id}, motion ID: {motion_id}, eigenvalue: {eigenvalue}, angle: {angle}, frame_row_size: {len(frame_row)}")
                print(frame_row.iloc[22:60])

                # 读取运动组合比例（3个分量）
                roll_pct = frame_row.iloc[base_col + 3] if i < 3 else 0
                pitch_pct = frame_row.iloc[base_col + 4] if i < 3 else 0
                yaw_pct = frame_row.iloc[base_col + 5] if i < 3 else 0
                
                if i >= 3:  # 平移方向
                    x_pct = frame_row.iloc[base_col + 3]
                    y_pct = frame_row.iloc[base_col + 4]
                    z_pct = frame_row.iloc[base_col + 5]
                    motion_comp = [0, 0, 0, x_pct, y_pct, z_pct]
                else:  # 旋转方向
                    motion_comp = [roll_pct, pitch_pct, yaw_pct, 0, 0, 0]
                
                # 标签
                if i < 3:
                    eigen_labels.append(r'$\mathbf{r_' + str(i) + '}$')
                    is_deg = frame_row[f'deg_rot{i+1}'] > 0
                else:
                    eigen_labels.append(r'$\mathbf{t_' + str(i-3) + '}$')
                    is_deg = frame_row[f'deg_trans{i-3+1}'] > 0
                
                motion_compositions.append(motion_comp)
                angles.append(angle)
                is_degenerate.append(is_deg)
                print(f"Frame {frame_id}, Motion {i}: Angle = {angle}, Is Degenerate = {is_deg}")
            else:
                print(f"Warning: Frame {frame_id} does not have enough data for motion {i}")
        
        # 转换为numpy数组
        motion_compositions = np.array(motion_compositions).T  # 6x6 -> 6x6 转置
        
        # 创建堆叠条形图
        x = np.arange(len(eigen_labels))
        width = 0.65
        
        # 绘制堆叠条形图
        bottom = np.zeros(len(eigen_labels))
        for j, (comp_row, label) in enumerate(zip(motion_compositions, motion_labels)):
            bars = ax.bar(x, comp_row, width, bottom=bottom, 
                          label=label if idx == 0 else None,  # 只在第一个子图创建图例
                          color=motion_colors[j], 
                          alpha=0.95, edgecolor='white', linewidth=1.5)
            
            # 收集图例句柄（只从第一个子图）
            if idx == 0 and j < 6:
                legend_handles.append(bars[0])
            
            # 添加百分比标签（只显示>10%的）
            for k, (height, btm) in enumerate(zip(comp_row, bottom)):
                if height > 10:
                    ax.text(k, btm + height/2, f'{height:.1f}', 
                           ha='center', va='center', fontsize=24, fontweight='bold')
            
            bottom += comp_row
        
        # 在r2和t0之间添加分隔线
        ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)
        
        # 在每个条形顶部添加角度信息和退化标记
        for j, (x_pos, angle, is_deg) in enumerate(zip(x, angles, is_degenerate)):
            # 使用更专业的标记：实心圆表示退化，空心圆表示非退化
            if is_deg:
                marker = '●'
                color = '#E74C3C'  # 红色
            else:
                marker = '○'
                color = '#1A1A1A'  # 深灰色

            ax.text(x_pos, 102, f'{angle:.1f}$\degree$', 
                   ha='center', va='bottom', fontsize=24)
            ax.text(x_pos, 108, marker, 
                   ha='center', va='bottom', fontsize=24, color=color)
        
        # 设置轴属性
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(eigen_labels, fontsize=28, fontfamily='Arial')
        ax.tick_params(axis='x', which='major', pad=10)
        ax.tick_params(axis='y', which='major', pad=5)
        
        # 设置y轴标签和Frame ID
        if idx == 0:
            ax.set_xticklabels([])
            ax.set_ylabel('Contribution Rate (%)', fontsize=26, labelpad=0)
            # 右侧添加Frame ID
            ax2 = ax.twinx()
            ax2.set_ylabel(f'Frame {frame_id+1}', fontsize=30, fontweight='bold', rotation=270, labelpad=30)
            ax2.set_yticks([])
        else:
            ax.set_ylabel('Contribution Rate (%)', fontsize=26, labelpad=0)
            # 右侧添加Frame ID
            ax2 = ax.twinx()
            ax2.set_ylabel(f'Frame {frame_id+1}', fontsize=30, fontweight='bold', rotation=270, labelpad=30)
            ax2.set_yticks([])
        
        # 添加网格
        ax.yaxis.grid(True, linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)
        
        # 设置边框颜色
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
        for spine in ax2.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)
    
    # 添加统一的图例到两个子图的正中心上方
    # 创建退化标记的图例项
    
    # 创建空白占位符用于布局
    empty = Line2D([0], [0], color='none', label='')
    
    # 创建退化标记
    # degenerate_marker = Line2D([0], [0], marker='●', color='w', 
    #                           markerfacecolor='#E74C3C', markersize=16, 
    #                           label='Degenerate', linestyle='None')
    # non_degenerate_marker = Line2D([0], [0], marker='○', color='w', 
    #                               markerfacecolor='#1A1A1A', markersize=16, 
    #                               label='Non-degenerate', linestyle='None')
    
    # # 按照要求的顺序排列图例：第一行Roll/Pitch/Yaw/标记，第二行X/Y/Z
    # first_row = [legend_handles[0], legend_handles[1], legend_handles[2], degenerate_marker]
    # second_row = [legend_handles[3], legend_handles[4], legend_handles[5], non_degenerate_marker]
    
    # first_row = [legend_handles[0], legend_handles[1], legend_handles[2]]
    # second_row = [legend_handles[3], legend_handles[4], legend_handles[5]]

    # all_handles = first_row + second_row
    # all_labels = ['Roll', 'Pitch', 'Yaw', 'Degenerate', 'X', 'Y', 'Z', 'Non-degenerate']
    # all_labels = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']

        # '●' (实心圆): 使用 marker='o' 并设置填充色
    degenerate_marker = Line2D([0], [0], marker='o', color='w', 
                              markerfacecolor='#E74C3C', markersize=16, 
                              label='D', linestyle='None')
    
    # '○' (空心圆): 使用 marker='o', 设置填充色为白色，边缘色为目标颜色
    non_degenerate_marker = Line2D([0], [0], marker='o', color='w', 
                                  markerfacecolor='w', markeredgecolor='#1A1A1A', markersize=16, 
                                  label='ND', linestyle='None')
    # --- 错误修复结束 ---

    # 按照要求的顺序排列图例：第一行Roll/Pitch/Yaw/标记，第二行X/Y/Z
    # first_row = [legend_handles[0], legend_handles[1], legend_handles[2], degenerate_marker]
    # second_row = [legend_handles[3], legend_handles[4], legend_handles[5], non_degenerate_marker]
    # 按照要求的顺序排列图例：第一行Roll/Pitch/Yaw/标记，第二行X/Y/Z
    first_row = [legend_handles[0], legend_handles[3], legend_handles[1],  legend_handles[4]]
    second_row = [legend_handles[2], legend_handles[5], degenerate_marker, non_degenerate_marker]


    all_handles = first_row + second_row
    # all_labels = ['Roll', 'Pitch', 'Yaw', 'Degen.', 'X', 'Y', 'Z', 'Non-degen.']
    all_labels = ['Roll', 'X', 'Pitch', 'Y', 'Yaw', 'Z', 'D', 'ND']    
    
    
    # 在图的顶部中心创建图例
    fig.legend(handles=all_handles, labels=all_labels, 
              loc='upper center', bbox_to_anchor=(0.54, 1.01),
              ncol=4, fontsize=22, frameon=True, 
              edgecolor='black', borderpad=0.8)
    
    # 调整布局以给图例留出空间
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # 保存图表
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved eigen direction combination to {output_path}")

def plot_eigen_direction_combination_fixed(dcreg_data, frame_ids=None, output_path='eigen_direction_combination.pdf'):
    """
    修正点：
    1) 退化标记按子空间 λ 升序对齐到当前 6 根柱；
    2) 仅对“退化轴的主成分段”进行美观强调：同色系对比描边（依据柱色自动选取深/浅对比色）+ 轻微白色晕圈；
    3) 图例中对应的主成分 Label 同步高亮（加粗并用同色系对比色着色）。
    其余（配色、顺序、布局、文字等）保持不变。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import colorsys

    # ---------------- helpers ----------------
    def stroke_color_for(face_hex):
        """依据柱色返回高对比描边色（同色系深/浅对比），确保不同底色上都清晰。"""
        r, g, b = mcolors.to_rgb(face_hex)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        # 浅色柱用更深的同色；深色柱用更亮的同色
        l2 = 0.32 if l >= 0.55 else 0.82
        r2, g2, b2 = colorsys.hls_to_rgb(h, l2, min(1.0, s * 1.05))
        return (r2, g2, b2)

    def make_proxy(face_hex, lw=1.5, edge='white'):
        return mpatches.Patch(facecolor=face_hex, edgecolor=edge, linewidth=lw)

    if frame_ids is None:
        frame_ids = FRAME_IDS_FOR_COMBINATION
    if len(frame_ids) > 2: frame_ids = frame_ids[:2]
    elif len(frame_ids) < 2:
        print("Need at least 2 frame IDs for comparison")
        return
    if dcreg_data.shape[1] < 60:
        print(f"Warning: DCReg data has only {dcreg_data.shape[1]} columns, expected at least 60")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 14))
    fig.subplots_adjust(hspace=0.15)

    # —— 保留你的配色与标签顺序 ——（索引：0..5 => Roll, Pitch, Yaw, X, Y, Z）
    motion_colors = {
        0: '#FF8C42',   # Roll
        1: '#5B9BD5',   # Pitch
        2: '#FDD0A2',   # Yaw
        3: '#2E75B6',   # X
        4: '#FFB366',   # Y
        5: '#9DC3E6'    # Z
    }
    motion_labels = ['Roll', 'Pitch', 'Yaw', 'X', 'Y', 'Z']

    # 用于图例高亮：收集两帧里出现过的“退化主成分层”索引（0..5）
    highlight_layers_global = set()

    # ---------- draw two frames ----------
    legend_handles = []

    for idx, (frame_id, ax) in enumerate(zip(frame_ids, axes)):
        if frame_id >= len(dcreg_data):
            print(f"Frame {frame_id} out of range"); 
            continue
        frame_row = dcreg_data.iloc[frame_id]

        # ===== 解析 alignment（顺序：旋转3 + 平移3）=====
        eigen_labels, angles, lambdas, comps = [], [], [], []
        for i in range(6):
            base = 22 + i * 6
            if base + 5 >= len(frame_row):
                print(f"Warning: Frame {frame_id} does not have enough data for motion {i}")
                continue

            eig = float(frame_row.iloc[base + 1])
            ang = float(frame_row.iloc[base + 2])

            if i < 3:  # 旋转：R/P/Y
                r = float(frame_row.iloc[base + 3])
                p = float(frame_row.iloc[base + 4])
                y = float(frame_row.iloc[base + 5])
                comp = [r, p, y, 0.0, 0.0, 0.0]
                eigen_labels.append(r'$\mathbf{r_' + str(i) + '}$')
            else:      # 平移：X/Y/Z
                x_ = float(frame_row.iloc[base + 3])
                y_ = float(frame_row.iloc[base + 4])
                z_ = float(frame_row.iloc[base + 5])
                comp = [0.0, 0.0, 0.0, x_, y_, z_]
                eigen_labels.append(r'$\mathbf{t_' + str(i-3) + '}$')

            lambdas.append(eig)
            comps.append(comp)
            angles.append(ang)

        # 退化 mask（命名列或按位置）
        def get_val(sr, name, pos):
            try:    return float(sr[name])
            except: return float(sr.iloc[pos])

        deg_rot   = [get_val(frame_row, 'deg_rot1', 10),   get_val(frame_row, 'deg_rot2', 11),   get_val(frame_row, 'deg_rot3', 12)]
        deg_trans = [get_val(frame_row, 'deg_trans1', 13), get_val(frame_row, 'deg_trans2', 14), get_val(frame_row, 'deg_trans3', 15)]

        # 将 mask 按子空间 λ 升序对齐到当前列
        is_degenerate = [False]*6
        rot_perm = list(np.argsort(lambdas[0:3]))
        for k in range(3):
            if deg_rot[k] > 0: is_degenerate[rot_perm[k]] = True
        trans_perm_local = list(np.argsort(lambdas[3:6]))
        for k in range(3):
            if deg_trans[k] > 0: is_degenerate[3 + trans_perm_local[k]] = True

        # ===== 绘条形（保留原样）=====
        motion_compositions = np.array(comps).T  # 6×6
        x = np.arange(len(eigen_labels))
        width = 0.65
        bottom = np.zeros(len(eigen_labels))
        bar_rects = [[None for _ in range(6)] for _ in range(6)]  # [层 j][列 k]

        for j, (comp_row, label) in enumerate(zip(motion_compositions, motion_labels)):
            bars = ax.bar(x, comp_row, width, bottom=bottom,
                          label=label if idx == 0 else None,
                          color=motion_colors[j],
                          alpha=0.95, edgecolor='white', linewidth=1.5)
            if idx == 0 and j < 6:
                legend_handles.append(bars[0])
            for k, rect in enumerate(bars):
                bar_rects[j][k] = rect
            for k, (h, btm) in enumerate(zip(comp_row, bottom)):
                if h > 10:
                    ax.text(k, btm + h/2, f'{h:.1f}',
                            ha='center', va='center', fontsize=24, fontweight='bold')
            bottom += comp_row

        # 分隔线 r2|t0
        ax.axvline(x=2.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.6)

        # 顶部角度与退化 ●/○（保留）
        for j, (x_pos, angle, is_deg) in enumerate(zip(x, angles, is_degenerate)):
            ax.text(x_pos, 102, f'{angle:.1f}$\\degree$',
                    ha='center', va='bottom', fontsize=24)
            ax.text(x_pos, 108, '●' if is_deg else '○',
                    ha='center', va='bottom', fontsize=24,
                    color=('#E74C3C' if is_deg else '#1A1A1A'))

        # ===== 仅强调“退化柱的主成分段”：同色系对比描边 + 轻微白色晕圈 =====
        for k in range(6):
            if not is_degenerate[k]:
                continue
            cand_layers = [0,1,2] if k < 3 else [3,4,5]
            vals = [motion_compositions[jj, k] for jj in cand_layers]
            jj_best = cand_layers[int(np.argmax(vals))]
            rect = bar_rects[jj_best][k]

            base_hex = motion_colors[jj_best]
            edge_col = stroke_color_for(base_hex)

            # 轻微白色晕圈（底层）
            glow = mpatches.FancyBboxPatch(
                (rect.get_x(), rect.get_y()),
                rect.get_width(), rect.get_height(),
                boxstyle="round,pad=0.03,rounding_size=0.05",
                linewidth=1.0, edgecolor=(1,1,1,0.85),
                facecolor='none', zorder=5
            )
            # 同色系对比描边（顶层）
            outline = mpatches.FancyBboxPatch(
                (rect.get_x(), rect.get_y()),
                rect.get_width(), rect.get_height(),
                boxstyle="round,pad=0.012,rounding_size=0.045",
                linewidth=4.0, edgecolor=edge_col,
                facecolor='none', zorder=6
            )
            ax.add_patch(glow)
            ax.add_patch(outline)

            # 记录用于“图例高亮”的主成分层
            highlight_layers_global.add(jj_best)

        # 轴/网格/边框（保留）
        ax.set_ylim(0, 115)
        ax.set_xticks(x)
        ax.set_xticklabels(eigen_labels, fontsize=28, fontfamily='Arial')
        ax.tick_params(axis='x', which='major', pad=10)
        ax.tick_params(axis='y', which='major', pad=5)
        ax.yaxis.grid(True, linestyle='--', alpha=0.2)
        ax.set_axisbelow(True)

        if idx == 0:
            ax.set_xticklabels([])
            ax.set_ylabel('Contribution Rate (%)', fontsize=26, labelpad=0)
            ax2 = ax.twinx(); ax2.set_ylabel(f'Frame {frame_id+1}', fontsize=30, fontweight='bold', rotation=270, labelpad=30); ax2.set_yticks([])
        else:
            ax.set_ylabel('Contribution Rate (%)', fontsize=26, labelpad=0)
            ax2 = ax.twinx(); ax2.set_ylabel(f'Frame {frame_id+1}', fontsize=30, fontweight='bold', rotation=270, labelpad=30); ax2.set_yticks([])

        for spine in ax.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(1.5)
        for spine in ax2.spines.values(): spine.set_edgecolor('black'); spine.set_linewidth(1.5)

    # ---------- Legend（保留原顺序，并对主成分层高亮） ----------
    # 先构造 6 个 proxy（不影响柱本身外观）
    base_proxies = [make_proxy(motion_colors[j], lw=1.5, edge='white') for j in range(6)]
    # 对需要高亮的层，替换为加粗+同色系对比描边的 proxy
    for j in list(highlight_layers_global):
        base_proxies[j] = make_proxy(motion_colors[j], lw=2.8, edge=stroke_color_for(motion_colors[j]))

    # 维持你原来的两行顺序：
    # 第一行：Roll, X, Pitch, Y
    # 第二行：Yaw, Z, D, ND
    degenerate_marker = Line2D([0], [0], marker='o', color='w',
                               markerfacecolor='#E74C3C', markersize=16,
                               label='D', linestyle='None')
    non_degenerate_marker = Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor='w', markeredgecolor='#1A1A1A', markersize=16,
                                   label='ND', linestyle='None')

    # 组合顺序与标签
    first_row  = [base_proxies[0], base_proxies[3], base_proxies[1], base_proxies[4]]
    second_row = [base_proxies[2], base_proxies[5], degenerate_marker, non_degenerate_marker]
    all_handles = first_row + second_row
    all_labels  = ['Roll', 'X', 'Pitch', 'Y', 'Yaw', 'Z', 'D', 'ND']

    leg = fig.legend(handles=all_handles, labels=all_labels,
                     loc='upper center', bbox_to_anchor=(0.54, 1.01),
                     ncol=4, fontsize=22, frameon=True,
                     edgecolor='black', borderpad=0.8)

    # 同步高亮图例文字（加粗+同色系对比色）
    # 组件索引 j -> 图例文本索引 idx
    comp_to_legend_text_idx = {0:0, 3:1, 1:2, 4:3, 2:4, 5:5}
    texts = leg.get_texts()
    for j in highlight_layers_global:
        t_idx = comp_to_legend_text_idx[j]
        texts[t_idx].set_fontweight('bold')
        texts[t_idx].set_color(stroke_color_for(motion_colors[j]))

    # 保存
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved eigen direction combination to {output_path}")



def plot_multi_dataset_comparison(dataset_configs, output_path='multi_dataset_comparison.pdf'):
    """
    绘制多数据集的退化检测比例对比图
    """
    # 收集所有数据集的统计信息
    all_stats = []
    
    for config in dataset_configs:
        print(f"Processing dataset: {config['name']}")
        
        # 加载数据
        dcreg_data = load_icp_data(config['dcreg'])
        fcn_data = load_icp_data(config['fcn'])
        me_data = load_icp_data(config['me'])
        
        if dcreg_data is None or fcn_data is None or me_data is None:
            print(f"  Skipping {config['name']} due to loading error")
            continue
        
        # 确保数据长度一致
        min_length = min(len(dcreg_data), len(fcn_data), len(me_data))
        dcreg_data = dcreg_data.iloc[:min_length]
        fcn_data = fcn_data.iloc[:min_length]
        me_data = me_data.iloc[:min_length]
        
        # 计算整体退化率
        deg_cols = ['deg_rot1', 'deg_rot2', 'deg_rot3', 'deg_trans1', 'deg_trans2', 'deg_trans3']
        
        # DCReg
        dcreg_rate = (dcreg_data[deg_cols].sum().sum() / (len(dcreg_data) * 6)) * 100
        
        # FCN
        fcn_rate = (fcn_data[deg_cols].sum().sum() / (len(fcn_data) * 6)) * 100
        
        # ME（基于特征值）
        eigen_cols = ['eigen1', 'eigen2', 'eigen3', 'eigen4', 'eigen5', 'eigen6']
        me_degenerate_count = 0
        for eigen_col in eigen_cols:
            me_degenerate_count += np.sum(me_data[eigen_col] < ME_EIGENVALUE_THRESHOLD)
        me_rate = (me_degenerate_count / (len(me_data) * 6)) * 100
        
        all_stats.append({
            'dataset': config['name'],
            'Ours': dcreg_rate,
            'FCN': fcn_rate,
            'ME': me_rate
        })
    
    if not all_stats:
        print("No valid datasets found!")
        return
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # 准备数据
    datasets = [stat['dataset'] for stat in all_stats]
    dcreg_rates = [stat['Ours'] for stat in all_stats]
    fcn_rates = [stat['FCN'] for stat in all_stats]
    me_rates = [stat['ME'] for stat in all_stats]
    
    # 设置条形图参数
    x = np.arange(len(datasets))
    width = 0.25
    
    # 绘制条形图
    bars1 = ax.bar(x - width, dcreg_rates, width, label='Ours',
                    color=COLOR_DCREG, alpha=0.9, edgecolor='white', linewidth=1)
    bars2 = ax.bar(x, fcn_rates, width, label='FCN',
                    color=COLOR_FCN, alpha=0.9, edgecolor='white', linewidth=1)
    bars3 = ax.bar(x + width, me_rates, width, label='ME',
                    color=COLOR_ME, alpha=0.9, edgecolor='white', linewidth=1)
    
    # 添加数值标签
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0.5:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=24,
                    fontweight='bold')
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    # 设置轴标签
    ax.set_ylabel('Overall Degeneracy Rate (%)', fontsize=30)
    ax.set_yticks([])  # 不显示y轴刻度
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=30)
    # how to add pad between the xticklabels and x aixs
    ax.tick_params(axis='x', which='major', pad=15)

    ax.legend(loc='upper center', frameon=True, fancybox=False, 
              shadow=False, fontsize=24, ncol=3, edgecolor='black', borderpad=0.8)
    
    # 设置y轴范围
    max_rate = max(max(dcreg_rates), max(fcn_rates), max(me_rates))
    ax.set_ylim(0, max_rate * 1.2)
    
    # 添加网格
    ax.yaxis.grid(True, linestyle='--', alpha=0.2)
    ax.set_axisbelow(True)
    
    # 设置边框颜色
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1.5)
    
    # 保存图表
    plt.savefig(output_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved multi-dataset comparison to {output_path}")

def generate_improved_summary_statistics(dcreg_data, fcn_data, me_data, output_path='summary_statistics.txt'):
    """
    生成改进的统计摘要
    """
    with open(output_path, 'w') as f:
        f.write("ICP Degeneracy Detection Methods - Summary Statistics\n")
        f.write("=" * 60 + "\n\n")
        
        # 条件数统计（DCReg）
        f.write("DCReg Condition Number Statistics:\n")
        f.write("-" * 40 + "\n")
        
        # 旋转条件数
        f.write("Rotational Condition Numbers:\n")
        f.write(f"  Schur Complement:\n")
        f.write(f"    Mean: {dcreg_data['cond_schur_rot'].mean():.2e}\n")
        f.write(f"    Std: {dcreg_data['cond_schur_rot'].std():.2e}\n")
        f.write(f"    Max: {dcreg_data['cond_schur_rot'].max():.2e}\n")
        f.write(f"    Frames > threshold: {np.sum(dcreg_data['cond_schur_rot'] > CONDITION_NUMBER_THRESHOLD)}\n")
        f.write(f"  Diagonal:\n")
        f.write(f"    Mean: {dcreg_data['cond_diag_rot'].mean():.2e}\n")
        f.write(f"    Std: {dcreg_data['cond_diag_rot'].std():.2e}\n")
        f.write(f"    Max: {dcreg_data['cond_diag_rot'].max():.2e}\n\n")
        
        # 平移条件数
        f.write("Translational Condition Numbers:\n")
        f.write(f"  Schur Complement:\n")
        f.write(f"    Mean: {dcreg_data['cond_schur_trans'].mean():.2e}\n")
        f.write(f"    Std: {dcreg_data['cond_schur_trans'].std():.2e}\n")
        f.write(f"    Max: {dcreg_data['cond_schur_trans'].max():.2e}\n")
        f.write(f"    Frames > threshold: {np.sum(dcreg_data['cond_schur_trans'] > CONDITION_NUMBER_THRESHOLD)}\n")
        f.write(f"  Diagonal:\n")
        f.write(f"    Mean: {dcreg_data['cond_diag_trans'].mean():.2e}\n")
        f.write(f"    Std: {dcreg_data['cond_diag_trans'].std():.2e}\n")
        f.write(f"    Max: {dcreg_data['cond_diag_trans'].max():.2e}\n\n")
        
        # 退化检测统计
        f.write("Degeneracy Detection Statistics:\n")
        f.write("-" * 40 + "\n")
        
        # DCReg统计
        deg_cols = ['deg_rot1', 'deg_rot2', 'deg_rot3', 'deg_trans1', 'deg_trans2', 'deg_trans3']
        total_frames = len(dcreg_data)
        
        f.write("\nDCReg (r0-r2: rotation, t0-t2: translation):\n")
        degenerate_frames = dcreg_data['is_degenerate'].sum()
        f.write(f"  Total degenerate frames: {degenerate_frames}/{total_frames} "
                f"({(degenerate_frames/total_frames)*100:.1f}%)\n")
        f.write("  Per-direction degeneracy rates:\n")
        for i, col in enumerate(deg_cols):
            label = ['$\mathbf{r_0}$', '$\mathbf{r_1}$', '$\mathbf{r_2}$', '$\mathbf{t_0}$', '$\mathbf{t_1}$', '$\mathbf{t_2}$'][i]
            rate = (dcreg_data[col].sum() / total_frames) * 100
            f.write(f"    {label}: {rate:.1f}%\n")
        
        # FCN统计
        f.write("\nFCN (v0-v5: cannot distinguish rot/trans):\n")
        degenerate_frames = fcn_data['is_degenerate'].sum()
        f.write(f"  Total degenerate frames: {degenerate_frames}/{total_frames} "
                f"({(degenerate_frames/total_frames)*100:.1f}%)\n")
        f.write("  Per-direction degeneracy rates:\n")
        for i, col in enumerate(deg_cols):
            label = f'v{i}'
            rate = (fcn_data[col].sum() / total_frames) * 100
            f.write(f"    {label}: {rate:.1f}%\n")
        
        # ME统计
        f.write(f"\nME (v0-v5: eigenvalue threshold = {ME_EIGENVALUE_THRESHOLD}):\n")
        eigen_cols = ['eigen1', 'eigen2', 'eigen3', 'eigen4', 'eigen5', 'eigen6']
        total_deg = 0
        f.write("  Per-direction degeneracy rates:\n")
        for i, eigen_col in enumerate(eigen_cols):
            label = f'v{i}'
            deg_count = np.sum(me_data[eigen_col] < ME_EIGENVALUE_THRESHOLD)
            rate = (deg_count / total_frames) * 100
            total_deg += deg_count
            f.write(f"    {label} (λ{i+1} < {ME_EIGENVALUE_THRESHOLD}): {rate:.1f}%\n")
        
        overall_rate = (total_deg / (total_frames * 6)) * 100
        f.write(f"  Overall degeneracy rate: {overall_rate:.1f}%\n")
    
    print(f"Saved improved summary statistics to {output_path}")

def main():
    """
    主函数：加载数据并生成所有图表
    """
    print("Loading ICP data files...")
    
    # 加载数据
    dcreg_data = load_icp_data(dcreg_file)
    fcn_data = load_icp_data(fcn_file)
    me_data = load_icp_data(me_file)
    
    # 检查数据是否成功加载
    if dcreg_data is None or fcn_data is None or me_data is None:
        print("Error: Failed to load one or more data files.")
        return
    
    print(f"Loaded DCReg data: {len(dcreg_data)} frames")
    print(f"Loaded FCN data: {len(fcn_data)} frames")
    print(f"Loaded ME data: {len(me_data)} frames")
    
    # 确保所有数据集长度一致
    min_length = min(len(dcreg_data), len(fcn_data), len(me_data))
    dcreg_data = dcreg_data.iloc[:min_length]
    fcn_data = fcn_data.iloc[:min_length]
    me_data = me_data.iloc[:min_length]
    
    print(f"\nGenerating visualizations for {min_length} frames...")
    print(f"ME eigenvalue threshold: {ME_EIGENVALUE_THRESHOLD}")
    print(f"Condition number threshold: {CONDITION_NUMBER_THRESHOLD}")
    
    # 生成图表
    plot_dcreg_condition_number_comparison(dcreg_data)
    plot_degeneracy_timeline_improved(dcreg_data, fcn_data, me_data)
    plot_degeneracy_ratio_comparison_single(dcreg_data, fcn_data, me_data)
    # plot_eigen_direction_combination(dcreg_data, FRAME_IDS_FOR_COMBINATION)
    plot_eigen_direction_combination_fixed(dcreg_data, FRAME_IDS_FOR_COMBINATION)

    # 如果有多个数据集配置，生成多数据集对比图
    if len(DATASET_CONFIGS) > 1:
        print("\nGenerating multi-dataset comparison...")
        plot_multi_dataset_comparison(DATASET_CONFIGS)
    
    generate_improved_summary_statistics(dcreg_data, fcn_data, me_data)
    
    print("\nAll visualizations completed successfully!")
    print("Generated files:")
    print("  - condition_number_comparison.pdf")
    print("  - degeneracy_detection_timeline.pdf")
    print("  - degeneracy_ratio_comparison.pdf")
    print("  - eigen_direction_combination.pdf")
    if len(DATASET_CONFIGS) > 1:
        print("  - multi_dataset_comparison.pdf")
    print("  - summary_statistics.txt")

if __name__ == "__main__":
    main()