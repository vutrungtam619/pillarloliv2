import numpy as np
import cv2
from configs.config import config

def random_flip_fusion(data_dict, flip_prob=0.5):
    if np.random.rand() >= flip_prob:
        return data_dict
    
    data_dict['pts'][:, 1] *= -1
    gt_bboxes = data_dict['gt_bboxes_3d']
    gt_bboxes[:, 1] *= -1
    gt_bboxes[:, 6] *= -1
    image = data_dict['image']
    data_dict['image'] = cv2.flip(image, 1)
    P = data_dict['calib_info']['P']
    w = image.shape[1]
    P[0, 2] = w - P[0, 2]
    P[0, 3] *= -1
    
    return data_dict

def color_jitter_fusion(data_dict, brightness=(0.8, 1.2), contrast=(0.8, 1.2), saturation=(0.8, 1.2), prob=0.5):
    if np.random.rand() > prob:
        return data_dict
    
    image = data_dict['image'].astype(np.float32) / 255.0
    b = np.random.uniform(*brightness) if brightness else 1.0
    c = np.random.uniform(*contrast) if contrast else 1.0
    s = np.random.uniform(*saturation) if saturation else 1.0
    image *= b
    mean = image.mean(axis=(0, 1), keepdims=True)
    image = (image - mean) * c + mean
    gray = image.mean(axis=2, keepdims=True)
    image = (image - gray) * s + gray
    data_dict['image'] = np.clip(image * 255, 0, 255).astype(np.uint8)
    return data_dict

def random_point_dropout(data_dict, max_dropout_ratio=0.1, prob=0.5):
    if np.random.rand() > prob:
        return data_dict

    pts = data_dict['pts']
    N = pts.shape[0]
    dropout_ratio = np.random.uniform(0, max_dropout_ratio)
    mask = np.random.rand(N) > dropout_ratio
    data_dict['pts'] = pts[mask]
    return data_dict

def point_jitter(data_dict, sigma=0.01, clip=0.02, prob=0.2):
    if np.random.rand() > prob:
        return data_dict

    pts = data_dict['pts']
    N = pts.shape[0]
    jitter = np.clip(sigma * np.random.randn(N, 3), -clip, clip)
    pts[:, :3] += jitter
    return data_dict

def points_shuffle(data_dict):
    np.random.shuffle(data_dict['pts'])
    return data_dict

def train_data_aug(data_dict):
    data_dict = random_flip_fusion(data_dict)
    data_dict = color_jitter_fusion(data_dict)
    data_dict = random_point_dropout(data_dict)
    data_dict = point_jitter(data_dict)
    data_dict = points_shuffle(data_dict)
    return data_dict

def val_data_aug(data_dict):
    return data_dict