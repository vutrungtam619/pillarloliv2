from pathlib import Path
root = root = Path(__file__).parent.parent

config = {
    'classes': {'Pedestrian': 0},
    'num_classes': 3,
    'data_root': 'kitti',
    'point_cloud_range': [0, -20.48, -4, 30.72, 20.48, 1],
    'voxel_size': [0.12, 0.16, 5],
    'max_voxels': (10000, 10000),
    'max_points': 32,
    'num_blocks': 10,
    'mean': [0.353464, 0.363416, 0.347248],
    'std': [0.295344, 0.304507, 0.322300],
    'shape': (1280, 720),
    'checkpoint_dir': Path(root) / 'checkpoints',
    'log_dir': Path(root) / 'logs',
    'batch_size_train': 8,
    'batch_size_val': 4,
    'num_workers': 4,
    'init_lr': 0.00025,
    'epoch': 20,
    'ckpt_freq': 2,
    'log_freq': 25,
}