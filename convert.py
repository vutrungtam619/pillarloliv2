import os
import json
import numpy as np
import shutil
import math
from pathlib import Path

# ==== Camera calibration parameters (KITTI setup) ====
P = np.array([
    [683.8, 0, 673.5907, 0.0],
    [0.0, 684.147, 372.8048, 0.0],
    [0, 0, 1, 0]
])
R_rect = np.eye(4)  # rectification = identity
T_lidar2cam = np.array([
    [0.00852965, -0.999945, -0.00606215, 0.0609592],
    [-0.0417155, 0.00570127, -0.999113, -0.144364],
    [0.999093, 0.00877497, -0.0416646, -0.0731114],
    [0, 0, 0, 1]
])


def create_kitti_directories(root_dir, splits=['training', 'validating']):
    root_path = Path(root_dir)
    root_path.mkdir(exist_ok=True)

    created_dirs = {}
    for split in splits:
        created_dirs[split] = {}
        for folder in ['image', 'label', 'velodyne']:
            dir_path = root_path / split / folder
            dir_path.mkdir(parents=True, exist_ok=True)
            created_dirs[split][folder] = str(dir_path)

    print("Making directories in ST_KITTI...")
    return created_dirs


def process_velodyne(input_file, output_file, pc_range=[0.01, -20.48, -3, 30.72, 20.48, 1]):
    points = np.fromfile(input_file, dtype=np.float32).reshape(-1, 4)
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &  # x
        (points[:, 1] >= y_min) & (points[:, 1] <= y_max) &  # y
        (points[:, 2] >= z_min) & (points[:, 2] <= z_max)    # z
    )
    filtered_points = points[mask]
    filtered_points.astype(np.float32).tofile(output_file)
    return None


def process_bbox2d(bbox2d):
    half_width, half_height = (bbox2d['w'] / 2), (bbox2d['h'] / 2)
    x_min = bbox2d['x'] - half_width
    x_max = bbox2d['x'] + half_width
    y_min = bbox2d['y'] - half_height
    y_max = bbox2d['y'] + half_height
    return x_min, y_min, x_max, y_max


def lidar_to_camera_box(cx, cy, cz, h, w, l, yaw_lidar):
    """Convert 3D box from LiDAR to Camera coordinates following KITTI format."""
    # --- Transform center ---
    center_lidar = np.array([cx, cy, cz, 1]).reshape(4, 1)
    center_cam = T_lidar2cam @ center_lidar

    # KITTI location is at bottom center
    center_cam[1, 0] -= h / 2.0

    # --- Rotation ---
    R = T_lidar2cam[:3, :3]
    dir_lidar = np.array([math.cos(yaw_lidar), math.sin(yaw_lidar), 0.0])
    dir_cam = R @ dir_lidar
    yaw_cam = math.atan2(dir_cam[0], dir_cam[2])  # rotation_y in KITTI

    return center_cam[0, 0], center_cam[1, 0], center_cam[2, 0], yaw_cam


def process_split(split, dir_path, root_path):
    anno_folder = Path(root_path) / "anno"  # Path to folder anno
    index = 0
    for sequence in split:
        anno_file = Path(anno_folder) / f"{sequence}.json"

        with open(anno_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"Process sequence {sequence}! Total number of frames is: {data['total_number']}")

        for frame in data['frames']:
            velodyne_input = Path(root_path) / frame['frame_name']
            image_input = Path(root_path) / frame['images'][0]['image_name']

            # --- Point cloud ---
            velodyne_output = Path(dir_path['velodyne']) / f"{index:06d}.bin"
            velodyne_output.touch(exist_ok=True)
            process_velodyne(velodyne_input, velodyne_output)

            # --- Image ---
            image_output = Path(dir_path['image'])
            shutil.copy(image_input, image_output / f"{index:06d}.jpg")

            # --- Label ---
            label_output = Path(dir_path['label']) / f"{index:06d}.txt"
            label_output.touch(exist_ok=True)

            person_items = [item for item in frame['items'] if item.get('category') == 'person']
            for item in person_items:
                item_id = item['id']
                person = next((it for it in frame['images'][0]['items'] if it['id'] == item_id), None)
                if person is None:
                    continue

                # 2D bbox
                bbox2d = {
                    'x': person['boundingbox']['x'],
                    'y': person['boundingbox']['y'],
                    'w': person['dimension']['x'],
                    'h': person['dimension']['y'],
                }

                # 3D bbox (LiDAR coord)
                bbox3d = {
                    'x': item['position']['x'],
                    'y': item['position']['y'],
                    'z': item['position']['z'],
                    'length': item['boundingbox']['y'],
                    'width': item['boundingbox']['x'],
                    'height': item['boundingbox']['z'],
                    'rotation': item['rotation']
                }

                # Convert to KITTI
                x_min, y_min, x_max, y_max = process_bbox2d(bbox2d)
                cx, cy, cz, rot_y = lidar_to_camera_box(
                    bbox3d['x'], bbox3d['y'], bbox3d['z'],
                    bbox3d['height'], bbox3d['width'], bbox3d['length'],
                    bbox3d['rotation']
                )

                # Order in KITTI: h, w, l
                h, w, l = bbox3d['height'], bbox3d['width'], bbox3d['length']

                # KITTI fields
                type_name = "Pedestrian"
                truncated = 0
                occluded = item['occlusion']
                alpha = 0  # can compute from proj if needed

                line = (
                    f'{type_name} '
                    f'{truncated} '
                    f'{occluded} '
                    f'{alpha:.6f} '
                    f'{x_min:.6f} {y_min:.6f} {x_max:.6f} {y_max:.6f} '
                    f'{h:.6f} {w:.6f} {l:.6f} '
                    f'{cx:.6f} {cy:.6f} {cz:.6f} '
                    f'{rot_y:.6f}\n'
                )

                with label_output.open('a', encoding='utf-8') as f:
                    f.write(line)

            index += 1
    print(f"Total of frame in this set is {index}")


def main():
    stcrowd_root = r"C:\SOURCE CODE\pillarloli\STCrowd"
    stkitti_root = r"C:\SOURCE CODE\pillarloli\STKitti"
    dir_path = create_kitti_directories(root_dir=stkitti_root)

    with open(Path(stcrowd_root, 'split.json'), 'r') as f:
        split_data = json.load(f)

    train_split, val_split = split_data['train'], split_data['val']

    print("Process train data into kitti! ...")
    process_split(train_split, dir_path['training'], stcrowd_root)
    print("Process val data into kitti! ...")
    process_split(val_split, dir_path['validating'], stcrowd_root)


if __name__ == '__main__':
    main()
