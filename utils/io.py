import pickle
import json
import numpy as np
from pathlib import Path

def read_points(file_path, dim=4, suffix='.bin'):
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        raise ValueError(f"File must be {suffix}, got {file_path.suffix}")
    return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)

def write_points(file_path, data: np.ndarray):
    file_path = Path(file_path)
    data.tofile(file_path)
        
def read_pickle(file_path, suffix='.pkl'):
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        raise ValueError(f"File must be {suffix}, got {file_path.suffix}")
    with file_path.open('rb') as f:
        return pickle.load(f)

def write_pickle(file_path, results):
    file_path = Path(file_path)
    with file_path.open('wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        
def read_calib(file_path):
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        'P': np.array(data["p"], dtype=float),
        'R': np.array(data["r"], dtype=float),
        'T': np.array(data["t"], dtype=float),
    }
    
def read_label(file_path, suffix=".txt"):
    """ Read label file, each line is one object in image """
    file_path = Path(file_path)
    if file_path.suffix != suffix:
        raise ValueError(f"File must be {suffix}, got {file_path.suffix}")

    text = file_path.read_text(encoding="utf-8").strip()
    if not text:  # file rỗng
        return {
            "names": np.array([], dtype=str),
            "truncated": np.array([], dtype=np.float32),
            "occluded": np.array([], dtype=np.int32),
            "alpha": np.array([], dtype=np.float32),
            "bbox": np.zeros((0, 4), dtype=np.float32),
            "dimensions": np.zeros((0, 3), dtype=np.float32),
            "locations": np.zeros((0, 3), dtype=np.float32),
            "rotation_y": np.array([], dtype=np.float32),
        }

    lines = [line.split() for line in text.splitlines()]
    return {
        "names": np.array([l[0] for l in lines]),
        "truncated": np.array([l[1] for l in lines], dtype=np.float32),
        "occluded": np.array([l[2] for l in lines], dtype=np.int32),
        "alpha": np.array([l[3] for l in lines], dtype=np.float32),
        "bbox": np.array([l[4:8] for l in lines], dtype=np.float32),
        "dimensions": np.array([l[8:11] for l in lines], dtype=np.float32)[:, [2, 0, 1]], # HWL -> LHW
        "locations": np.array([l[11:14] for l in lines], dtype=np.float32),
        "rotation_y": np.array([l[14] for l in lines], dtype=np.float32),
    }

def write_label(file_path, result):
    file_path = Path(file_path)
    with file_path.open("w", encoding="utf-8") as f:
        num_objects = len(result["names"])
        if num_objects == 0:
            return  # file rỗng, không ghi gì
        for i in range(num_objects):
            line = (
                f'{result["names"][i]} '
                f'{result["truncated"][i]} '
                f'{result["occluded"][i]} '
                f'{result["alpha"][i]} '
                f'{" ".join(map(str, result["bbox"][i]))} '
                f'{" ".join(map(str, result["dimensions"][i]))} '
                f'{" ".join(map(str, result["locations"][i]))} '
                f'{result["rotation_y"][i]} '
                f'{result.get("scores", ["-1"] * num_objects)[i]}\n'
            )
            f.write(line)
