import argparse
import os
import cv2
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from configs.config import config
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import (
    read_calib,
    read_points,
    read_label,
    write_points,
    write_pickle,
    remove_outside_points,
    remove_outside_bboxes,
    get_points_num_in_bbox,
)

root = Path(__file__).parent

def process_one_idx(idx, data_root, split, label, lidar_reduced_folder):
    cur_info_dict = {}

    image_path = Path(data_root) / split / 'image' / f'{idx}.jpg'
    lidar_path = Path(data_root) / split / 'velodyne' / f'{idx}.bin'
    calib_path = Path(data_root) / split / 'calib.json'

    cur_info_dict['index'] = idx

    image = cv2.imread(str(image_path))
    image_shape = image.shape[:2]
    cur_info_dict['image'] = {
        'image_shape': image_shape,
        'image_path': Path(*image_path.parts[-3:]) # ví dụ: training/image/000000.jpg
    }
    
    calib_dict = read_calib(calib_path)
    cur_info_dict['calib'] = calib_dict
    
    lidar_reduced_path = Path(lidar_reduced_folder) / f'{idx}.bin'
    lidar_points = read_points(lidar_path)
    reduced_points = remove_outside_points(
        points=lidar_points,
        r0_rect=calib_dict['R'],
        tr_velo_to_cam=calib_dict['T'],
        P2=calib_dict['P'],
        image_shape=image_shape
    )
    write_points(lidar_reduced_path, reduced_points)
    cur_info_dict['lidar'] = {
        'lidar_total': reduced_points.shape[0],
        'lidar_path': lidar_reduced_path,
    }
    
    if label:
        label_path = Path(data_root) / split / 'label' / f'{idx}.txt'
        annotation_dict = read_label(label_path)
        annotation_dict = remove_outside_bboxes(
            annotation_dict,
            r0_rect=calib_dict['R'],
            tr_velo_to_cam=calib_dict['T'],
            P2=calib_dict['P'],
            image_shape=image_shape
        )
        annotation_dict['num_points_in_gt'] = get_points_num_in_bbox(
            points=reduced_points,
            r0_rect=calib_dict['R'],
            tr_velo_to_cam=calib_dict['T'],
            dimensions=annotation_dict['dimensions']  ,
            locations=annotation_dict['locations']    ,
            rotation_y=annotation_dict['rot_z'],
            names=annotation_dict['names']  
        ) 
        cur_info_dict['annos'] = annotation_dict

    return int(idx), cur_info_dict


def create_data_info_pkl(data_root, data_type, label):
    print(f"Processing {data_type} data into pkl file....")
    
    index_files = Path(root) / 'index' / f'{data_type}.txt'
    ids = index_files.read_text(encoding="utf-8").splitlines()
    
    split = 'training' if label else 'testing'
    
    lidar_reduced_folder = Path(root) / 'datasets' / 'reduced'
    lidar_reduced_folder.mkdir(exist_ok=True)
    
    infos_dict = {}
    with ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_one_idx, idx, data_root, split, label, lidar_reduced_folder): idx
            for idx in ids
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, cur_info_dict = future.result()
            infos_dict[idx] = cur_info_dict
    
    save_pkl_path = Path(root) / 'datasets' / f'infos_{data_type}.pkl'
    write_pickle(save_pkl_path, infos_dict)  
    
    return None     
    
def main(args):
    data_root = args.data_root
    
    kitti_train_infos_dict = create_data_info_pkl(data_root, data_type='train', label=True)
    kitti_val_infos_dict = create_data_info_pkl(data_root, data_type='val', label=True)

    print("......Processing finished!!!")  
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--data_root', default=config['data_root'])
    args = parser.parse_args()

    main(args)