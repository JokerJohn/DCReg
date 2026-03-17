import argparse
import time
import os

arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


def str2bool(v):
    return v.lower() in ('true', '1')


dataset = '3DMatch'
experiment_id = f"PointDSC_{dataset}_{time.strftime('%m%d%H%M')}"
# snapshot configurations
snapshot_arg = add_argument_group('Snapshot')
snapshot_arg.add_argument('--snapshot_dir', type=str, default=f'snapshot/{experiment_id}')
snapshot_arg.add_argument('--tboard_dir', type=str, default=f'tensorboard/{experiment_id}')
snapshot_arg.add_argument('--snapshot_interval', type=int, default=1)
snapshot_arg.add_argument('--save_dir', type=str, default=os.path.join(f'snapshot/{experiment_id}', 'models/'))

# Network configurations
net_arg = add_argument_group('Network')
net_arg.add_argument('--in_dim', type=int, default=6)
net_arg.add_argument('--num_layers', type=int, default=12)
net_arg.add_argument('--num_channels', type=int, default=128)
net_arg.add_argument('--num_iterations', type=int, default=10, help='power iteration algorithm')
net_arg.add_argument('--ratio', type=float, default=0.1, help='max ratio of seeding points')
net_arg.add_argument('--k', type=int, default=40, help='size of local neighborhood')

# Loss configurations
loss_arg = add_argument_group('Loss')
loss_arg.add_argument('--evaluate_interval', type=int, default=1)
loss_arg.add_argument('--balanced', type=str2bool, default=False)
loss_arg.add_argument('--weight_classification', type=float, default=1.0)
loss_arg.add_argument('--weight_spectralmatching', type=float, default=1.0)
loss_arg.add_argument('--weight_transformation', type=float, default=0.0)
loss_arg.add_argument('--transformation_loss_start_epoch', type=int, default=0)

# Optimizer configurations
opt_arg = add_argument_group('Optimizer')
opt_arg.add_argument('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'])
opt_arg.add_argument('--max_epoch', type=int, default=50)
opt_arg.add_argument('--training_max_iter', type=int, default=3500)
opt_arg.add_argument('--val_max_iter', type=int, default=1000)
opt_arg.add_argument('--lr', type=float, default=1e-4)
opt_arg.add_argument('--weight_decay', type=float, default=1e-6)
opt_arg.add_argument('--momentum', type=float, default=0.9)
opt_arg.add_argument('--scheduler', type=str, default='ExpLR')
opt_arg.add_argument('--scheduler_gamma', type=float, default=0.99)
opt_arg.add_argument('--scheduler_interval', type=int, default=1)

# Dataset and dataloader configurations
data_arg = add_argument_group('Data')
if dataset == '3DMatch':
    data_arg.add_argument('--root', type=str, default='/data/3DMatch')
    data_arg.add_argument('--descriptor', type=str, default='fcgf', choices=['d3feat', 'fpfh', 'fcgf'])
    data_arg.add_argument('--inlier_threshold', type=float, default=0.10)
    net_arg.add_argument('--sigma_d', type=float, default=0.10)
    data_arg.add_argument('--downsample', type=float, default=0.03)
    data_arg.add_argument('--re_thre', type=float, default=15, help='rotation error thrshold (deg)')
    data_arg.add_argument('--te_thre', type=float, default=30, help='translation error thrshold (cm)')
else:
    data_arg.add_argument('--root', type=str, default='/data/KITTI')
    data_arg.add_argument('--descriptor', type=str, default='fcgf', choices=['fcgf', 'fpfh'])
    data_arg.add_argument('--inlier_threshold', type=float, default=1.2)
    net_arg.add_argument('--sigma_d', type=float, default=1.2)
    data_arg.add_argument('--downsample', type=float, default=0.30)
    data_arg.add_argument('--re_thre', type=float, default=5, help='rotation error thrshold (deg)')
    data_arg.add_argument('--te_thre', type=float, default=60, help='translation error thrshold (cm)')

data_arg.add_argument('--num_node', type=int, default=1000)
data_arg.add_argument('--use_mutual', type=str2bool, default=False)
data_arg.add_argument('--augment_axis', type=int, default=3)
data_arg.add_argument('--augment_rotation', type=float, default=1.0, help='rotation angle = num * 2pi')
data_arg.add_argument('--augment_translation', type=float, default=0.5, help='translation = num (m)')
data_arg.add_argument('--batch_size', type=int, default=16)
data_arg.add_argument('--num_workers', type=int, default=16)

# Other configurations
misc_arg = add_argument_group('Misc')
misc_arg.add_argument('--gpu_mode', type=str2bool, default=True)
misc_arg.add_argument('--verbose', type=str2bool, default=True)
misc_arg.add_argument('--pretrain', type=str, default='')
misc_arg.add_argument('--weights_fixed', type=str2bool, default=False)


def get_config():
    args = parser.parse_args()
    return args
