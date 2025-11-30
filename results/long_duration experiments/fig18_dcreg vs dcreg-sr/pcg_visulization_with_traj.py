
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from matplotlib import rcParams
# import zipfile
# import os
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.gridspec as gridspec

# # 设置Science/Nature期刊风格
# plt.style.use('seaborn-v0_8-whitegrid')
# rcParams['font.family'] = 'sans-serif'
# rcParams['font.sans-serif'] = ['Arial']
# rcParams['font.size'] = 16
# rcParams['axes.labelsize'] = 18
# rcParams['axes.titlesize'] = 20
# rcParams['legend.fontsize'] = 14
# rcParams['xtick.labelsize'] = 14
# rcParams['ytick.labelsize'] = 14
# rcParams['figure.dpi'] = 300
# rcParams['savefig.dpi'] = 300
# rcParams['lines.linewidth'] = 2.0
# rcParams['axes.linewidth'] = 1.0
# rcParams['axes.edgecolor'] = '#333333'
# rcParams['grid.alpha'] = 0.3
# rcParams['grid.color'] = '#cccccc'

# # Science/Nature配色方案 - 使用浅色系
# COLOR_PCG = '#4A90E2'          # 明亮蓝色 - PCG
# COLOR_QR = '#F5A623'           # 明亮橙色 - Projection
# COLOR_DEGENERATE_FILL = '#FFE5CC'  # 非常淡的橙色 - 退化区域
# COLOR_DEGENERATE_EDGE = '#FFA500'  # 橙色边界

# def load_ate_from_zip(ate_zip_path):
#     """从EVO生成的zip文件中加载ATE数据"""
#     with zipfile.ZipFile(ate_zip_path, 'r') as zip_ref:
#         zip_ref.extractall("temp_ate")
    
#     error_array = np.load('temp_ate/error_array.npy')
#     timestamps = np.load('temp_ate/timestamps.npy')
    
#     import shutil
#     shutil.rmtree('temp_ate')
    
#     return timestamps, error_array

# def load_pcg_data(filename='pcg.txt'):
#     """加载PCG数据"""
#     data = np.loadtxt(filename)
    
#     columns = ['timestamp', 'cond_H', 'cond_PH', 'cond_improvement_ratio',
#                'converged_iterations', 'time_pcg_ms', 'time_qr_direct_ms',
#                'first_iter_residual', 'first_iter_precond_residual', 
#                'first_iter_alpha', 'first_iter_rz_product',
#                'final_residual_pcg', 'final_residual_qr_direct',
#                'solution_diff_norm', 'degenerate_update_ratio',
#                'noise_amplification_factor', 'is_degenerate']
    
#     if data.shape[1] > len(columns):
#         for i in range(len(columns), data.shape[1]):
#             columns.append(f'extra_col_{i}')
    
#     df = pd.DataFrame(data[:, :len(columns)], columns=columns[:data.shape[1]])
#     df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
#     df['is_degenerate'] = df['is_degenerate'].astype(bool)
#     df['frame_id'] = np.arange(len(df))  # 添加frame id
    
#     return df

# def create_comprehensive_figure(df, ate_pcg_path, ate_qr_path, 
#                               poses_pcg_path, poses_qr_path, 
#                               gt_path, save_prefix='dcReg'):
#     """创建包含轨迹和性能分析的综合图表"""
    
#     # 加载数据
#     ate_time_pcg, ate_error_pcg = load_ate_from_zip(ate_pcg_path)
#     ate_time_qr, ate_error_qr = load_ate_from_zip(ate_qr_path)
    
#     # 加载轨迹数据
#     poses_pcg = pd.read_csv(poses_pcg_path, sep=" ", header=None)
#     poses_qr = pd.read_csv(poses_qr_path, sep=" ", header=None)
#     gt = pd.read_csv(gt_path, sep=" ", header=None)
    
#     # 提取坐标
#     n_points = min(len(ate_error_pcg), len(poses_pcg), len(poses_qr))
#     pcg_x = poses_pcg.iloc[:n_points, 1]
#     pcg_y = poses_pcg.iloc[:n_points, 2]
#     qr_x = poses_qr.iloc[:n_points, 1]
#     qr_y = poses_qr.iloc[:n_points, 2]
    
#     # 创建frame id对应的ATE
#     frame_ids = np.arange(n_points)
    
#     # 创建图形 - 使用GridSpec进行精确布局
#     fig = plt.figure(figsize=(14, 12))
#     gs = gridspec.GridSpec(3, 2, figure=fig, 
#                           height_ratios=[2, 2, 2],
#                           width_ratios=[2, 2],
#                           hspace=0.20, wspace=0.01)
    
#     # 第一行：两个轨迹子图
#     ax1_left = fig.add_subplot(gs[0, 0])
#     ax1_right = fig.add_subplot(gs[0, 1], sharey=ax1_left)
    
#     # 第二行：ATE对比
#     ax2 = fig.add_subplot(gs[1, :])
    
#     # 第三行：条件数对比
#     ax3 = fig.add_subplot(gs[2, :], sharex=ax2)
    
#     # 计算全局的颜色范围
#     vmin = min(ate_error_pcg.min(), ate_error_qr.min())
#     vmax = max(ate_error_pcg.max(), ate_error_qr.max())
    
#     # 子图1左：PCG轨迹
#     sc1 = ax1_left.scatter(pcg_x, pcg_y, c=ate_error_pcg[:n_points], 
#                           cmap='RdBu_r', s=8, alpha=0.8, 
#                           vmin=vmin, vmax=vmax)
#     ax1_left.scatter(pcg_x.iloc[0], pcg_y.iloc[0], color='green', 
#                     marker='^', s=200, label='Start', zorder=5, edgecolors='black')
#     ax1_left.scatter(pcg_x.iloc[-1], pcg_y.iloc[-1], color='red', 
#                     marker='v', s=200, label='End', zorder=5, edgecolors='black')
#     ax1_left.set_title('DCReg', fontsize=18, pad=10)
#     ax1_left.set_xlabel('X (m)', fontsize=16)
#     ax1_left.set_ylabel('Y (m)', fontsize=16)
#     ax1_left.grid(True, alpha=0.3)
#     ax1_left.set_aspect('equal', adjustable='box')
#     ax1_left.legend(loc='best', frameon=True, edgecolor='black', fontsize=12)
    
#     # 子图1右：Without PCG轨迹
#     sc2 = ax1_right.scatter(qr_x, qr_y, c=ate_error_qr[:n_points], 
#                            cmap='RdBu_r', s=8, alpha=0.8,
#                            vmin=vmin, vmax=vmax)
#     ax1_right.scatter(qr_x.iloc[0], qr_y.iloc[0], color='green', 
#                      marker='^', s=200, zorder=5, edgecolors='black')
#     ax1_right.scatter(qr_x.iloc[-1], qr_y.iloc[-1], color='red', 
#                      marker='v', s=200, zorder=5, edgecolors='black')
#     ax1_right.set_title('DCReg with SR', fontsize=18, pad=10)
#     ax1_right.set_xlabel('X (m)', fontsize=16)
#     ax1_right.grid(True, alpha=0.3)
#     ax1_right.set_aspect('equal', adjustable='box')
    
#     # 隐藏右侧子图的Y轴标签
#     plt.setp(ax1_right.get_yticklabels(), visible=False)
    
#     # 在右侧添加共享的colorbar
#     cbar_ax = fig.add_axes([0.92, 0.62, 0.02, 0.25])
#     cbar = fig.colorbar(sc2, cax=cbar_ax, orientation='vertical')
#     cbar.set_label('ATE (m)', fontsize=16)
#     cbar.ax.tick_params(labelsize=14)
    
#     # 子图2：ATE随frame id变化
#     mean_ate_qr = np.mean(ate_error_qr)
#     mean_ate_pcg = np.mean(ate_error_pcg)
    
#     ax2.plot(frame_ids, ate_error_qr[:n_points], color=COLOR_QR, 
#              label=f'SR (mean: {mean_ate_qr:.3f}m)', 
#              alpha=0.8, linewidth=2)
#     ax2.plot(frame_ids, ate_error_pcg[:n_points], color=COLOR_PCG, 
#              label=f'PCG (mean: {mean_ate_pcg:.3f}m)', 
#              alpha=0.9, linewidth=2.5)
    
#     # 退化区域填充
#     degenerate_regions = df['is_degenerate'][:n_points]
#     ax2.fill_between(frame_ids, 0, ate_error_qr[:n_points].max()*1.1, 
#                      where=degenerate_regions, alpha=0.2, 
#                      color=COLOR_DEGENERATE_FILL, 
#                      edgecolor=COLOR_DEGENERATE_EDGE, linewidth=0.5)
    
#     ax2.axhline(mean_ate_qr, color=COLOR_QR, linestyle='--', alpha=0.5, linewidth=1.5)
#     ax2.axhline(mean_ate_pcg, color=COLOR_PCG, linestyle='--', alpha=0.5, linewidth=1.5)
    
#     ax2.set_ylabel('ATE (m)', fontsize=18)
#     ax2.legend(loc='upper left', frameon=True, edgecolor='black', ncol=2,
#               fancybox=False, framealpha=1.0, fontsize=14)
#     ax2.grid(True, alpha=0.3)
#     ax2.set_xlim([0, n_points-1])
    
#     # 隐藏x轴标签（与下面的子图共享）
#     plt.setp(ax2.get_xticklabels(), visible=False)
    
#     # 子图3：条件数对比
#     ax3.semilogy(df['frame_id'], df['cond_H'], 
#                  color=COLOR_QR, label='Original $\kappa(\mathbf{H})$', 
#                  alpha=0.8, linewidth=2)
#     ax3.semilogy(df['frame_id'], df['cond_PH'], 
#                  color=COLOR_PCG, label='Preconditioned $\kappa(\mathbf{PH})$', 
#                  alpha=0.9, linewidth=2.5)
    
#     # 退化区域填充
#     ax3.fill_between(df['frame_id'], 1e0, df['cond_H'].max()*10, 
#                      where=df['is_degenerate'], alpha=0.2, 
#                      color=COLOR_DEGENERATE_FILL,
#                      edgecolor=COLOR_DEGENERATE_EDGE, linewidth=0.5,
#                      label='Degenerate Regions')
    
#     ax3.set_xlabel('Frame ID', fontsize=18)
#     ax3.set_ylabel('Condition Number', fontsize=18)
#     ax3.legend(loc='upper left', frameon=True, edgecolor='black',  ncol=3,
#               fancybox=False, framealpha=1.0, fontsize=14)
#     ax3.grid(True, alpha=0.3)
#     ax3.set_ylim([1e0, df['cond_H'].max()*30])
#     ax3.set_xlim([0, n_points-1])
    
#     # 设置所有子图的边框
#     for ax in [ax1_left, ax1_right, ax2, ax3]:
#         for spine in ax.spines.values():
#             spine.set_edgecolor('black')
#             spine.set_linewidth(1)
    
#     # 调整布局
#     plt.tight_layout()
    
#     # 保存图形
#     plt.savefig(f'{save_prefix}_comprehensive_analysis.pdf', 
#                 bbox_inches='tight', dpi=300)
#     # plt.show()
#     plt.close()
    
#     # 打印统计信息
#     print("\n=== Performance Summary ===")
#     print(f"ATE improvement: {(mean_ate_qr - mean_ate_pcg)/mean_ate_qr*100:.1f}%")
#     degenerate_mask = df['is_degenerate']
#     if degenerate_mask.sum() > 0:
#         print(f"Average condition improvement in degenerate regions: "
#               f"{df.loc[degenerate_mask, 'cond_improvement_ratio'].mean():.1f}×")
#     print(f"Degenerate frames: {degenerate_mask.sum()}/{len(df)} "
#           f"({degenerate_mask.sum()/len(df)*100:.1f}%)")

# # 主函数
# if __name__ == "__main__":
#     path = '/home/xchu/data/ltloc_result/parkinglot_10_all/parkinglot_raw_10_ours'
    
#     # 加载数据
#     df = load_pcg_data(f'{path}/pcg.txt')
    
#     # 创建综合图表
#     create_comprehensive_figure(df, 
#                                ate_pcg_path=f'{path}/pcg_ate.zip',
#                                ate_qr_path=f'{path}/qr_ate.zip',
#                                poses_pcg_path=f'{path}/optimized_poses_tum.txt',  # PCG轨迹
#                                poses_qr_path=f'{path}/optimized_poses_tum_qr.txt',   # Without PCG轨迹
#                                gt_path=f'{path}/Parkinglot-2023-10-28-18-59-01_0.005_ins_tum.txt',
#                                save_prefix='dcReg')


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
import zipfile
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

# 设置Science/Nature期刊风格
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'Arial'
rcParams['font.size'] = 30  # 基础字体大小改为20
# rcParams['axes.labelsize'] = 30
# rcParams['axes.titlesize'] = 30
# rcParams['legend.fontsize'] = 26
# rcParams['xtick.labelsize'] = 26
# rcParams['ytick.labelsize'] = 26
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['lines.linewidth'] = 2.5
rcParams['axes.linewidth'] = 1.5
rcParams['axes.edgecolor'] = '#333333'
rcParams['grid.alpha'] = 0.3
rcParams['grid.color'] = '#d0d0d0'

# Science/Nature学术配色方案 - 使用现代浅色系
COLOR_PCG = '#2E86AB'          # 深蓝色 - PCG主色
COLOR_QR = '#F24236'           # 橙红色 - Projection主色
COLOR_DEGENERATE_FILL = '#FFF3E0'  # 非常淡的橙色 - 退化区域填充
COLOR_DEGENERATE_EDGE = '#FFB74D'  # 浅橙色边界
COLOR_START = '#43A047'        # 绿色 - 起点
COLOR_END = '#E53935'          # 红色 - 终点

# 辅助颜色
COLOR_MEAN_LINE = '#666666'    # 灰色虚线
COLOR_GRID = '#E0E0E0'         # 淡灰色网格

def load_ate_from_zip(ate_zip_path):
    """从EVO生成的zip文件中加载ATE数据"""
    with zipfile.ZipFile(ate_zip_path, 'r') as zip_ref:
        zip_ref.extractall("temp_ate")
    
    error_array = np.load('temp_ate/error_array.npy')
    timestamps = np.load('temp_ate/timestamps.npy')
    
    import shutil
    shutil.rmtree('temp_ate')
    
    return timestamps, error_array

def load_pcg_data(filename='pcg.txt'):
    """加载PCG数据"""
    data = np.loadtxt(filename)
    
    columns = ['timestamp', 'cond_H', 'cond_PH', 'cond_improvement_ratio',
               'converged_iterations', 'time_pcg_ms', 'time_qr_direct_ms',
               'first_iter_residual', 'first_iter_precond_residual', 
               'first_iter_alpha', 'first_iter_rz_product',
               'final_residual_pcg', 'final_residual_qr_direct',
               'solution_diff_norm', 'degenerate_update_ratio',
               'noise_amplification_factor', 'is_degenerate']
    
    if data.shape[1] > len(columns):
        for i in range(len(columns), data.shape[1]):
            columns.append(f'extra_col_{i}')
    
    df = pd.DataFrame(data[:, :len(columns)], columns=columns[:data.shape[1]])
    df['time_sec'] = (df['timestamp'] - df['timestamp'].iloc[0]) / 1e9
    df['is_degenerate'] = df['is_degenerate'].astype(bool)
    df['frame_id'] = np.arange(len(df))
    
    return df

def create_comprehensive_figure(df, ate_pcg_path, ate_qr_path, 
                              poses_pcg_path, poses_qr_path, 
                              gt_path, save_prefix='dcReg'):
    """创建包含轨迹和性能分析的综合图表"""
    
    # 加载数据
    ate_time_pcg, ate_error_pcg = load_ate_from_zip(ate_pcg_path)
    ate_time_qr, ate_error_qr = load_ate_from_zip(ate_qr_path)
    
    # 加载轨迹数据
    poses_pcg = pd.read_csv(poses_pcg_path, sep=" ", header=None)
    poses_qr = pd.read_csv(poses_qr_path, sep=" ", header=None)
    gt = pd.read_csv(gt_path, sep=" ", header=None)
    
    # 提取坐标 - 交换X和Y轴以适应纵向轨迹
    n_points = min(len(ate_error_pcg), len(poses_pcg), len(poses_qr))
    pcg_x = poses_pcg.iloc[:n_points, 2]  # 原来的y作为x
    pcg_y = poses_pcg.iloc[:n_points, 1]  # 原来的x作为y
    qr_x = poses_qr.iloc[:n_points, 2]    # 原来的y作为x
    qr_y = poses_qr.iloc[:n_points, 1]    # 原来的x作为y
    
    # 创建frame id对应的ATE
    frame_ids = np.arange(n_points)
    
    # 创建图形 - 优化的GridSpec布局
    fig = plt.figure(figsize=(16, 20))
    # gs = gridspec.GridSpec(3, 3, figure=fig, 
    #                       height_ratios=[2, 2, 2],
    #                       width_ratios=[2.5, 0.3, 2.5],  # 中间留出空间给colorbar
    #                       hspace=0.25, wspace=0.05)
    # i want to set the height space between the ATE subfigure and the condition number subfigure to 0.2
    
     # 上部分：轨迹图
    gs_top = gridspec.GridSpec(1, 3, figure=fig, 
                              height_ratios=[1],
                              width_ratios=[2.0, 0.2, 2.0],
                              left=0.0, right=1.0, top=0.98, bottom=0.64, wspace=0.2)

    # 下部分：ATE和条件数图
    gs_bottom = gridspec.GridSpec(2, 1, figure=fig,
                                 height_ratios=[1, 1],
                                 left=0.0, right=1.0, top=0.62, bottom=0.0, hspace=0.1)


       # 第一行：两个轨迹子图
    ax1_left = fig.add_subplot(gs_top[0, 0])
    ax1_right = fig.add_subplot(gs_top[0, 2], sharey=ax1_left)
    
    # 第二行：ATE对比
    ax2 = fig.add_subplot(gs_bottom[0])
    
    # 第三行：条件数对比
    ax3 = fig.add_subplot(gs_bottom[1], sharex=ax2)

    # # 第一行：两个轨迹子图
    # ax1_left = fig.add_subplot(gs[0, 0])
    # ax1_right = fig.add_subplot(gs[0, 2], sharey=ax1_left)
    
    # # 第二行：ATE对比（跨越所有列）
    # ax2 = fig.add_subplot(gs[1, :])
    
    # # 第三行：条件数对比（跨越所有列）
    # ax3 = fig.add_subplot(gs[2, :], sharex=ax2)
    
    # 计算全局的颜色范围
    vmin = min(ate_error_pcg.min(), ate_error_qr.min())
    vmax = max(ate_error_pcg.max(), ate_error_qr.max())
    
    # 使用更适合的colormap - 从好（蓝）到差（红）
    cmap = 'bwr'
    
    # 子图1左：PCG轨迹
    sc1 = ax1_left.scatter(pcg_x, pcg_y, c=ate_error_pcg[:n_points], 
                          cmap=cmap, s=10, alpha=1.0, 
                          vmin=vmin, vmax=vmax, edgecolors='none')
    ax1_left.scatter(pcg_x.iloc[0], pcg_y.iloc[0], color=COLOR_START, 
                    marker='o', s=300, label='Start', zorder=5, 
                    edgecolors='white', linewidth=2)
    ax1_left.scatter(pcg_x.iloc[-1], pcg_y.iloc[-1], color=COLOR_END, 
                    marker='s', s=300, label='End', zorder=5, 
                    edgecolors='white', linewidth=2)
    ax1_left.set_title('DCReg', fontsize=32, pad=15, fontweight='bold')
    ax1_left.set_xlabel('Y (m)', fontsize=30)  # 注意标签交换
    ax1_left.set_ylabel('X (m)', fontsize=30)  # 注意标签交换
    ax1_left.tick_params(axis='both', which='major', pad=10)
    ax1_left.grid(True, alpha=0.3, color=COLOR_GRID)
    ax1_left.set_aspect('equal', adjustable='box')
    ax1_left.legend(loc='lower right', frameon=True, facecolor='white', 
                   edgecolor='black', fontsize=28, framealpha=0.9)
    
    # 子图1右：Without PCG轨迹
    sc2 = ax1_right.scatter(qr_x, qr_y, c=ate_error_qr[:n_points], 
                           cmap=cmap, s=10, alpha=1.0,
                           vmin=vmin, vmax=vmax, edgecolors='none')
    ax1_right.scatter(qr_x.iloc[0], qr_y.iloc[0], color=COLOR_START, 
                     marker='o', s=300, zorder=5, 
                     edgecolors='white', linewidth=2)
    ax1_right.scatter(qr_x.iloc[-1], qr_y.iloc[-1], color=COLOR_END, 
                     marker='s', s=300, zorder=5, 
                     edgecolors='white', linewidth=2)
    ax1_right.set_title('DCReg-SR', fontsize=32, pad=15, fontweight='bold')
    ax1_right.set_xlabel('Y (m)', fontsize=30)  # 注意标签交换
    ax1_right.grid(True, alpha=0.2, color=COLOR_GRID)
    ax1_right.set_aspect('equal', adjustable='box')
    ax1_right.tick_params(axis='x', which='major', pad=10)
    # 隐藏右侧子图的Y轴标签
    plt.setp(ax1_right.get_yticklabels(), visible=False)
    
    # 在中间添加colorbar
    cbar_ax = fig.add_subplot(gs_top[0, 1])
    # gs_top[0, 1] should set the position of the colorbar.especially the height

    cbar = fig.colorbar(sc2, cax=cbar_ax, orientation='vertical')
    # i want to add the label on the top center of the colorbar, how to do it?

    # 获取当前位置并调整高度
    pos = cbar_ax.get_position()
    # 设置新位置：保持x位置和宽度不变，调整y位置和高度
    # pos.x0, pos.y0 是左下角坐标，width和height是宽度和高度
    new_height = 0.25  # 设置为原高度的25%
    y_center = pos.y0 + pos.height/2  # 找到中心点
    new_y0 = y_center - new_height/2  # 计算新的底部位置
    cbar_ax.set_position([pos.x0, new_y0, pos.width, new_height])

    cbar.ax.set_title('ATE', fontsize=32, pad=10)
    # cbar.set_label('ATE (m)', fontsize=22, labelpad=10)
    cbar.ax.tick_params(labelsize=26)
    # cbar.ax.tick_params("y", major=True, labelsize=22)
    cbar.ax.tick_params(axis='y', which='major', pad=5)
    # how to set the height of the colorbar to be 0.25?
    # cbar_ax.set_ylim([vmin, vmax])
    
    # 子图2：ATE随frame id变化
    mean_ate_qr = np.mean(ate_error_qr)
    mean_ate_pcg = np.mean(ate_error_pcg)
    
    # 退化区域填充（放在最底层）
    degenerate_regions = df['is_degenerate'][:n_points]
    ax2.fill_between(frame_ids, 0, ate_error_qr[:n_points].max()*1.1, 
                     where=degenerate_regions, alpha=0.8, 
                     color=COLOR_DEGENERATE_FILL, 
                     edgecolor='none', label='Degenerate Regions')
    
    # 绘制ATE曲线
    ax2.plot(frame_ids, ate_error_qr[:n_points], color=COLOR_QR, 
             label=f'SR ({mean_ate_qr:.3f}m)', 
             alpha=0.9, linewidth=3)
    ax2.plot(frame_ids, ate_error_pcg[:n_points], color=COLOR_PCG, 
             label=f'PCG ({mean_ate_pcg:.3f}m)', 
             alpha=0.9, linewidth=3)
    
    # 平均值虚线
    ax2.axhline(mean_ate_qr, color=COLOR_QR, linestyle='--', alpha=0.5, linewidth=3)
    ax2.axhline(mean_ate_pcg, color=COLOR_PCG, linestyle='--', alpha=0.5, linewidth=3)
    
    ax2.set_ylabel('ATE (m)', fontsize=32)
    ax2.legend(loc='upper left', frameon=True, facecolor='white', 
              edgecolor='black', ncol=3, framealpha=0.95, fontsize=28)
    ax2.grid(True, alpha=0.3, color=COLOR_GRID)
    ax2.set_xlim([0, n_points-1])
    ax2.set_ylim(bottom=0)
    ax2.tick_params(axis='y', which='major', pad=5)

    
    # 隐藏x轴标签（与下面的子图共享）
    plt.setp(ax2.get_xticklabels(), visible=False)
    
    # 子图3：条件数对比
    # 退化区域填充（放在最底层）
    ax3.fill_between(df['frame_id'], 1e0, df['cond_H'].max()*10, 
                     where=df['is_degenerate'], alpha=0.8, 
                     color=COLOR_DEGENERATE_FILL,
                     edgecolor='none')
    
    ax3.semilogy(df['frame_id'], df['cond_H'], 
                 color=COLOR_QR, label='Original $\kappa(\mathbf{H})$', 
                 alpha=0.9, linewidth=3)
    ax3.semilogy(df['frame_id'], df['cond_PH'], 
                 color=COLOR_PCG, label='Preconditioned $\kappa(\mathbf{PH})$', 
                 alpha=0.9, linewidth=3)
    
    ax3.set_xlabel('Frame ID', fontsize=32)
    ax3.set_ylabel('Cond.', fontsize=32)
    ax3.legend(loc='upper left', frameon=True, facecolor='white',
              edgecolor='black', ncol=2, framealpha=0.95, fontsize=28)
    ax3.grid(True, alpha=0.3, color=COLOR_GRID, which='both')
    ax3.set_ylim([1e0, df['cond_H'].max()*30])
    ax3.set_xlim([0, n_points-1])
    ax3.tick_params(axis='x', which='major', pad=10)
    ax3.tick_params(axis='y', which='major', pad=5)

    
    # 设置所有子图的边框为更细的线条
    for ax in [ax1_left, ax1_right, ax2, ax3]:
        for spine in ax.spines.values():
            spine.set_edgecolor("#000000")
            spine.set_linewidth(1.5)
            spine.set_alpha(1.0)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图形
    plt.savefig(f'{save_prefix}_comprehensive_analysis.pdf', 
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.savefig(f'{save_prefix}_comprehensive_analysis.png', 
                bbox_inches='tight', dpi=300, facecolor='white')
    # plt.show()
    plt.close()
    
    # 打印统计信息
    print("\n=== Performance Summary ===")
    print(f"ATE improvement: {(mean_ate_qr - mean_ate_pcg)/mean_ate_qr*100:.1f}%")
    degenerate_mask = df['is_degenerate']
    if degenerate_mask.sum() > 0:
        print(f"Average condition improvement in degenerate regions: "
              f"{df.loc[degenerate_mask, 'cond_improvement_ratio'].mean():.1f}×")
    print(f"Degenerate frames: {degenerate_mask.sum()}/{len(df)} "
          f"({degenerate_mask.sum()/len(df)*100:.1f}%)")

# 主函数
if __name__ == "__main__":
    path = '/home/xchu/data/ltloc_result/parkinglot_10_all/parkinglot_raw_10_ours'
    
    # 加载数据
    df = load_pcg_data(f'{path}/pcg.txt')
    
    # 创建综合图表
    create_comprehensive_figure(df, 
                               ate_pcg_path=f'{path}/pcg_ate.zip',
                               ate_qr_path=f'{path}/qr_ate.zip',
                               poses_pcg_path=f'{path}/optimized_poses_tum.txt',
                               poses_qr_path=f'{path}/optimized_poses_tum_qr.txt',
                               gt_path=f'{path}/Parkinglot-2023-10-28-18-59-01_0.005_ins_tum.txt',
                               save_prefix='dcReg')