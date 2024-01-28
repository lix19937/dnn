from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch

import torch.utils.data as data
import numpy as np
import math
import json
import collections
from datasets.sample.lidar_seg import align_cloud_size
from utils.image import gaussian_radius, draw_umich_gaussian
# from . import preprocess_cpp

import time


class LidarOd(data.Dataset):
    def __init__(self):
        super(LidarOd, self).__init__()
        self.global_id = collections.OrderedDict()

    def _alpha_to_8(self, alpha):
        ret = [0, 0, 0, 1, 0, 0, 0, 1]
        if alpha < np.pi / 6. or alpha > 5 * np.pi / 6.:
            r = alpha - (-0.5 * np.pi)
            ret[1] = 1
            ret[2], ret[3] = np.sin(r), np.cos(r)
        if alpha > -np.pi / 6. or alpha < -5 * np.pi / 6.:
            r = alpha - (0.5 * np.pi)
            ret[5] = 1
            ret[6], ret[7] = np.sin(r), np.cos(r)
        return ret

    def _convert_alpha(self, alpha):
        return math.radians(alpha + 45) if self.alpha_in_degree else alpha

    def _read_calibration(self, label):
        calibration = label['calibration']
        rot_mat = calibration['rotation_matrix']
        rot_mat = np.array(rot_mat, dtype=np.float32)
        yaw = float(calibration['rotation_angle'][0])
        trans_mat = calibration['translation_matrix']
        trans_mat = np.array(trans_mat, dtype=np.float32)
        return rot_mat, yaw, trans_mat

    def _parse_lidar_cloud(self, data_file, calibration):
        lidar_one_frame = np.fromfile(data_file, dtype=np.float32)
        lidar_one_frame = lidar_one_frame.reshape(-1, 3)
        assert calibration is not None
        lidar_one_frame = lidar_one_frame @ calibration['rot'].T + \
            calibration['trans']
        return lidar_one_frame

    def _parse_label_results_lidar128(self, label_file):
        with open(label_file, 'r') as f:
            data = json.load(f)
        # Reads calibration.
        rot_mat, yaw, trans_mat = self._read_calibration(data)

        data = data['label']['3D']
        parsed_results = []
        for one_obj in data:
            if 'Light_info' in one_obj.keys():
                print('Fake obj with Light_info is ignored.')
                continue
            id = one_obj['globalID']
            if id not in self.global_id:
                self.global_id[id] = len(self.global_id)
            center_lidar = np.array([float(one_obj['position']['x']), float(
                one_obj['position']['y']), float(one_obj['position']['z'])], dtype=np.float32)
            center_ego = center_lidar @ rot_mat.T + trans_mat
            parsed_obj = {
                'id': self.global_id[id],
                'center': center_ego.tolist(),
                'heading': float(one_obj['rotation']['phi']) + yaw,
                'dim': [float(one_obj['size'][0]), float(one_obj['size'][1]), float(one_obj['size'][2])]}
            category = self.class_128_to_luminar[one_obj['type']] \
                if one_obj['type'] in self.class_128_to_luminar else 'Background'
            parsed_obj['category'] = category
            parsed_obj['category_id'] = self.class_to_id[parsed_obj['category']]

            parsed_results.append(parsed_obj)
        calibration = {'rot': rot_mat, 'trans': trans_mat}
        return parsed_results, calibration

    def _parse_label_results_luminar(self, label_file):
        with open(label_file, 'r') as f:
            data = json.load(f)

        # Reads calibration.
        rot_mat, yaw, trans_mat = self._read_calibration(data)

        objects = data['Objects']
        parsed_results = []
        for one_obj in objects:
            center_lidar = np.array([
                float(one_obj['center'][0]), float(one_obj['center'][1]), float(one_obj['center'][2])],
                dtype=np.float32)
            center_ego = center_lidar @ rot_mat.T + trans_mat
            parsed_obj = {
                'id': int(one_obj['ID']),
                'center': center_ego.tolist(),
                'heading': float(one_obj['heading']) + yaw,
                'dim': [float(one_obj['dim'][0]), float(one_obj['dim'][1]), float(one_obj['dim'][2])]}
            # Typo in labeled results.
            category = one_obj['cateory']
            if category not in self.class_name:
                print('Wrong type in labeled results:', one_obj['cateory'])
                category = 'Background'
            parsed_obj['category'] = category
            parsed_obj['category_id'] = self.class_to_id[parsed_obj['category']]

            parsed_results.append(parsed_obj)
        calibration = {'rot': rot_mat, 'trans': trans_mat}
        return parsed_results, calibration

    def _augment(self, cloud, labels):
        theta = np.random.uniform(-1.0, 1.0) * np.pi / 18.0  # +/- 10deg.
        trans_x = np.random.uniform(-1.0, 1.0) * 1.0
        trans_y = np.random.uniform(-1.0, 1.0) * 1.0
        trans_z = np.random.uniform(-1.0, 1.0) * 0.1

        sin_a = np.sin(theta)
        cos_a = np.cos(theta)
        # Aug points.
        cloud = torch.from_numpy(cloud)
        cloud = preprocess_cpp.augment(
            cloud, sin_a, cos_a, trans_x, trans_y, trans_z)
        cloud = cloud.numpy()
        # Aug boxes.
        for idx in range(len(labels)):
            x = labels[idx]['center'][0] * cos_a - \
                labels[idx]['center'][1] * sin_a + trans_x
            y = labels[idx]['center'][0] * sin_a + \
                labels[idx]['center'][1] * cos_a + trans_y
            labels[idx]['center'][0] = x
            labels[idx]['center'][1] = y
            labels[idx]['center'][2] += trans_z
            labels[idx]['heading'] += theta

        return cloud, labels

    def __getitem__(self, index):
        # Reads labels and calibration.
        calibration = None
        label_path = self.labels[index]
        if self.lidar_type == 'luminar':
            labels, calibration = self._parse_label_results_luminar(label_path)
        else:
            labels, calibration = self._parse_label_results_lidar128(
                label_path)
        # Reads lidar data.
        data_path = self.images[index]
        lidar_one_frame = self._parse_lidar_cloud(
            data_path, calibration=calibration)

        # Augmentation.
        aug_prob = np.random.uniform(0.0, 1.0)
        if self.split == 'train' and aug_prob < self.opt.aug_lidar:
            lidar_one_frame, labels = self._augment(lidar_one_frame, labels)

        # Transforms cloud to bev feature map, which is input for backbone.
        # Python too slow. Uses cpp extension instead.
        lidar_one_frame = torch.from_numpy(lidar_one_frame)
        inp = preprocess_cpp.build(
            lidar_one_frame,
            self.channels_num,
            self.opt.input_h,
            self.opt.input_w,
            self.roi_x_min,
            self.roi_y_min,
            self.min_z,
            self.max_z,
            self.bev_res,
            self.height_res)
        inp = inp.numpy()
        inp[0, :, :] /= self.roi_x_max
        inp[1, :, :] /= self.roi_y_max
        inp[2:, :, :] = np.log(inp[2:, :, :] + 1.0)

        num_classes = self.num_classes
        hm = np.zeros((num_classes, self.opt.output_h,
                      self.opt.output_w), dtype=np.float32)
        # wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        rot = np.zeros((self.max_objs, 2), dtype=np.float32)   # sin, cos
        dim = np.zeros((self.max_objs, 3), dtype=np.float32)
        height = np.zeros((self.max_objs, 1), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        rot_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cls_ids = np.full((self.max_objs), -1, dtype=np.int)

        num_objs = min(len(labels), self.max_objs)
        draw_gaussian = draw_umich_gaussian
        gt_det = []
        for k in range(num_objs):
            ann = labels[k]
            cls_id = ann['category_id']
            if cls_id < 0:
                continue
            l = ann['dim'][0] / self.bev_res / self.opt.down_ratio
            w = ann['dim'][1] / self.bev_res / self.opt.down_ratio
            if l > 0. and w > 0.:
                radius = gaussian_radius(
                    (l, w), min_overlap=self.opt.gaussian_overlap)
                # Large radius has positive affect on recall.
                radius = max(self.opt.min_radius, int(radius))
                radius += 8
                ct = np.array(
                    [(ann['center'][0] - self.roi_x_min) / self.bev_res / self.opt.down_ratio,
                     (ann['center'][1] - self.roi_y_min) / self.bev_res / self.opt.down_ratio], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                if not (0 <= ct_int[0] < self.opt.output_h and 0 <= ct_int[1] < self.opt.output_w):
                    continue
                # h0 * W + w0.
                ind_tmp = int(ct_int[0] * self.opt.output_w + ct_int[1])
                draw_gaussian(hm[cls_id], ct, radius)
                gt_det.append([ct[0], ct[1], 1] +
                              self._alpha_to_8(self._convert_alpha(ann['heading'])) +
                              ann['center'] + ann['dim'] + [cls_id])
                heading = self._convert_alpha(ann['heading'])
                rot[k] = np.sin(heading), np.cos(heading)
                dim[k] = np.log(ann['dim'])
                height[k] = ann['center'][2]
                ind[k] = ind_tmp
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                if ann['category'] not in self.no_heading_class:
                    rot_mask[k] = 1
                cls_ids[k] = cls_id
        ret = {
            'input': inp,
            'hm': hm,
            'dim': dim,
            'z': height,
            'ind': ind,
            'rot': rot,
            'reg_mask': reg_mask,
            'rot_mask': rot_mask,
            'cls_id': cls_ids}
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not ('train' in self.split):
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 18), dtype=np.float32)
            meta = {'gt_det': gt_det,
                    'image_path': data_path, 'timestamp': data_path.split('/')[-1][:-4]}
            if calibration is not None:
                meta['calib'] = calibration
            ret['meta'] = meta
        # For segmentation.
        ret['aligned_cloud'] = align_cloud_size(lidar_one_frame.numpy().transpose(), self.opt.align_size)
        ret['indices'] = np.zeros((1, self.opt.align_size), np.int32)
        ret['mask'] = np.zeros((1, self.opt.align_size), np.int32)
        return ret
