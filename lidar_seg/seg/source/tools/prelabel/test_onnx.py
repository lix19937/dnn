from numpy import dtype
import onnxruntime
import numpy as np
import torch
import argparse
import open3d
import preprocess_cpp


transform = np.eye(3)
transform[0, 2] = 0
transform[1, 2] = 76.8
transform = transform / 0.2
aligned_size = 100000

def read_pcd(pcd_path):
    points = open3d.io.read_point_cloud(pcd_path)
    points = np.asarray(points.points, dtype=np.float32)
    return points


def read_calibration(path):
    from scipy.spatial.transform import Rotation as R

    with open(path, 'r') as f:
        lines = f.read().splitlines()
    lines = [_.split(',') for _ in lines]
    lines = [[_.strip() for _ in line] for line in lines]
    quat = [float(lines[6][1]), float(lines[7][0]), float(lines[7][1][:-2]), float(lines[6][0][8:])]
    translation = [float(lines[12][0][8:]), float(lines[12][1]), float(lines[13][0][:-2])]
    calibration = {}
    calibration['rotation_matrix'] = R.from_quat(quat).as_matrix().astype(np.float32)
    calibration['translation_matrix'] = translation
    return calibration


def points_lidar2vehicle(points, calibration):
    points = points @ calibration['rotation_matrix'].T + \
        calibration['translation_matrix']
    return points

def align_cloud_size(cloud, max_size=50000, constant_value=0):
    if cloud.shape[1] >= max_size:
        return cloud[:, :max_size, ...]
    else:
        return np.pad(cloud, ((0, 0), (0, max_size - cloud.shape[1])), constant_values=constant_value)

def project_to_bev(cloud):
    ori_cloud = cloud.copy()
    ori_cloud[:, -1] = 1
    trans_cloud = np.matmul(transform, ori_cloud.T)
    indices = trans_cloud[:2, :].astype(np.int32).T
    mask = np.zeros((1, indices.shape[0]), np.int32)
    dict_mat = dict()
    for idx in range(indices.shape[0]):
        tmp_tuple = tuple(indices[idx, :])
        if tmp_tuple not in dict_mat:
            dict_mat[tmp_tuple] = [idx]
        else:
            dict_mat[tmp_tuple].append(idx)

    for key, val in dict_mat.items():
        if 0 <= key[0] < 768 and 0 <= key[1] < 768:
            mask[0, val] = 1
            
    torch_cloud = torch.from_numpy(cloud)
    inp = preprocess_cpp.build(
        torch_cloud, 16, 768, 768, 0.0, -76.8,
        -1.2, 3.0, 0.2, 0.3)
    inp = inp.numpy()
    inp[0, :, :] /= 153.6
    inp[1, :, :] /= 76.8
    inp[2:, :, :] = np.log(inp[2:, :, :] + 1.0)

    indices = align_cloud_size((indices[:, 0] * 768 + indices[:, 1]).reshape(1, -1), aligned_size)
    mask = align_cloud_size(mask, aligned_size)
    results = dict()
    results['input'] = inp[None, ...]
    results['aligned_cloud'] = align_cloud_size(cloud.transpose(), aligned_size)[None, ...]
    results['indices'] = indices[None, ...]
    results['mask'] = mask[None, ...]
    return results


def main(onnx_path, cloud_path, calibration_path):
    print('onnxruntime version: ', onnxruntime.__version__)
    sess = onnxruntime.InferenceSession(
        onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    cloud = read_pcd(cloud_path)

    calibration = read_calibration(calibration_path)
    cloud = points_lidar2vehicle(cloud, calibration).astype(np.float32)
    # cloud = torch.from_numpy(cloud)
    results = project_to_bev(cloud)

    net_feed_input = ["input", "indices", "mask", "aligned_cloud"]
    mm_inputs = {"input": results['input'], "indices": results['indices'],
                 "mask": results['mask'], "aligned_cloud": results['aligned_cloud']}
    # mm_inputs = {"input": dummy_input}
    onnx_result = sess.run(
        None, {net_feed_input[0]: mm_inputs[net_feed_input[0]],
               net_feed_input[1]: mm_inputs[net_feed_input[1]],
               net_feed_input[2]: mm_inputs[net_feed_input[2]],
               net_feed_input[3]: mm_inputs[net_feed_input[3]]})[0]

    res_file_name = cloud_path.replace('.pcd', '.bin')
    out_size = min(cloud.shape[0], aligned_size)
    points = np.concatenate([cloud[:out_size, :], onnx_result[:, :out_size].reshape(-1, 1).astype(np.float32)], axis=1)
    points.tofile(res_file_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx_path', required=True, type=str)
    parser.add_argument('--cloud_path', required=True, type=str)
    parser.add_argument('--calibration_path', required=True, type=str)
    args = parser.parse_args()
    main(args.onnx_path, args.cloud_path, args.calibration_path)
