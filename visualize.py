import numpy as np
import open3d as o3d


def load_point_cloud(bin_path):
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, points


def load_labels(label_path):
    labels = []
    with open(label_path, "r") as f:
        for line in f.readlines():
            data = line.strip().split(" ")
            if len(data) < 15:
                continue
            obj_type = data[0]
            h, w, l = map(float, data[8:11])
            x, y, z = map(float, data[11:14])  # camera coords
            ry = float(data[14])  # rotation around Y (camera)
            labels.append({
                "type": obj_type,
                "h": h, "w": w, "l": l,
                "x": x, "y": y, "z": z,
                "ry": ry
            })
    return labels


def camera_to_lidar_box(label, Tr):
    h, w, l = label["h"], label["w"], label["l"]
    x, y, z = label["x"], label["y"], label["z"]
    ry = label["ry"]

    # Tâm box (Open3D cần tâm giữa)
    center_cam = np.array([x, y + h / 2, z, 1.0])

    # Rotation matrix quanh Y camera
    R_cam = np.array([
        [ np.cos(ry), 0, np.sin(ry)],
        [ 0,          1, 0],
        [-np.sin(ry), 0, np.cos(ry)]
    ])

    # Tính hướng (phía trước dọc theo Z camera sau khi quay ry)
    heading_cam = np.array([np.sin(ry), 0, np.cos(ry), 0.0])

    # Convert sang lidar
    center_lidar = (Tr @ center_cam.reshape(4,1)).flatten()[:3]
    heading_lidar = (Tr @ (center_cam + heading_cam)).flatten()[:3]
    R_lidar = Tr[:3, :3] @ R_cam

    return center_lidar, R_lidar, [l, h, w], heading_lidar


def create_3d_box(center, R, size):
    box = o3d.geometry.OrientedBoundingBox(center, R, size)
    box.color = (1, 0, 0)
    return box


def create_heading_arrow(center, heading, scale=1.5):
    # line set (center -> heading point)
    points = [center, center + (heading - center) * scale]
    lines = [[0, 1]]
    colors = [[0, 1, 0]]  # xanh lá
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def main(bin_path, label_path, calib):
    pcd, _ = load_point_cloud(bin_path)
    labels = load_labels(label_path)
    Tr_velo2cam = np.array(calib["t"], dtype=np.float64)   # lidar → camera
    Tr_cam2velo = np.linalg.inv(Tr_velo2cam)               # camera → lidar

    geoms = [pcd]
    for label in labels:
        center, R, size, heading = camera_to_lidar_box(label, Tr_cam2velo)
        obb = create_3d_box(center, R, size)
        arrow = create_heading_arrow(center, heading, scale=2.0)
        geoms.extend([obb, arrow])

    o3d.visualization.draw_geometries(geoms)



if __name__ == "__main__":
    calib = {
        "p": [[683.8,0,673.5907,0.0],[0.0,684.147,372.8048,0.0],[0,0,1,0]],
        "r": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
        "t": [
            [0.00852965,-0.999945,-0.00606215,0.0609592],
            [-0.0417155,0.00570127,-0.999113,-0.144364],
            [0.999093,0.00877497,-0.0416646,-0.0731114],
            [0,0,0,1]
        ]
    }

    bin_path = r"STKitti\training\velodyne\000000.bin"
    label_path = r"STKitti\training\label\000000.txt"
    main(bin_path, label_path, calib)
