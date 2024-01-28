# -*- coding: utf-8 -*-
# @Time    : 2022/7/29 下午4:28
# @Author  : Teanna
# @File    : saic.py
# @Software: PyCharm

import os
from loguru import logger

import cv2
import numpy as np
from pycocotools.coco import COCO

from ..dataloading import get_yolox_datadir
from .datasets_wrapper import Dataset

from .saic_classes import SAIC_CLASSES
from yolox.data.transforms import obb2poly_np_pure_oc

DEBUG = bool(int(os.getenv('DEBUG', 0)))


class SAICDataset(Dataset):
    """
    SAIC dataset class.
    """

    def __init__(self, data_dir=None, image_type='height', name="train", img_size=(416, 416), preproc=None, cache=False):
        """
        SAIC dataset initialization.
        Args:
            data_dir (str): dataset root directory
            data_type(str): data type (e.g. 'height', 'snr' or 'full')
            name (str): SAIC data name (e.g. 'train2017' or 'val2017')
            img_size (int): target image size after pre-processing
            preproc: data augmentation strategy
        """
        super().__init__(img_size)
        assert os.path.exists(data_dir), '{} not exists'.format(data_dir)

        self.name = name

        self.class_to_ind = dict(zip(SAIC_CLASSES, range(len(SAIC_CLASSES))))
        self.data_dir = data_dir
        self.img_dir = os.path.join(data_dir, name, 'image_%s' % image_type)
        self.label_dir = os.path.join(data_dir, name, 'label')
        self.label_files = os.listdir(self.label_dir)

        self.img_size = img_size
        self.preproc = preproc
        self.annotations = [self._load_anno_along_index(index) for index in range(self.__len__())]

    def __len__(self):
        return len(self.label_files)

    def _load_anno_along_index(self, index):
        """
        Return:
        res: the label information of per image, [n_gt_per_img, 9]
        9 contain [8 * polys of dets, labels]
        """

        file_name = self.label_files[index]
        res = np.loadtxt(os.path.join(self.label_dir, file_name), dtype=str)
        res = res.reshape(-1, res.shape[-1])
        res[..., 8:9] = np.apply_along_axis(lambda x: self.class_to_ind[x[0]], 0, res[..., 8:9])
        res = res[..., :9].astype(np.float)

        width = 500
        height = 500

        r = min(self.img_size[0] / width, self.img_size[1] / width)
        res[:, :8] *= r
        img_info = (height, width)
        resized_info = (int(height * r), int(width * r))
        return res, img_info, resized_info, file_name

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.img_dir, file_name[:-3] + 'png')

        img = cv2.imread(img_file)
        assert img is not None, f"file named {img_file} not found"

        return img

    def pull_item(self, index):
        res, img_info, resized_info, file_name = self.annotations[index]
        img = self.load_resized_img(index)

        return img, res.copy(), img_info, file_name[:-4]

    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 9]`.
                each label consists of [class, xc, yc, w, h]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): same as the input index. Used for evaluation.
        """
        img, target, img_info, img_id = self.pull_item(index)

        if DEBUG:
            frame_ori = img.astype(np.uint8)
            # frame_ori = img
            frame_ori = cv2.drawContours(frame_ori, target[:, :8].reshape(-1, 4, 1, 2).astype(np.int), -1, (0, 0, 255), 1)
            cv2.imshow('ori', frame_ori)

        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)

        if DEBUG:
            frame = img.transpose(1, 2, 0) * 255.0  # .astype(np.uint8)
            polys = obb2poly_np_pure_oc(target[..., 1:6])
            polys = polys.reshape(-1, 4, 1, 2)
            frame = cv2.drawContours(cv2.UMat(frame).get(), polys.astype(np.int), -1, (0, 0, 255), 1)
            cv2.imshow('debug', frame)
            if cv2.waitKey(0) == ord('q'):
                exit(0)
        # logger.info(target[np.sum(target, axis=-1) > 0])
        return img, target, img_info, img_id

    def evaluate_detection(self, data_list, save_results_dir=None, eval_online=False, eval_func=None):
        # if not eval_online:
        #     eval_online = self.name == 'val'
        # if is_submiss:
        #     assert not eval_online, "is_submiss and eval online can't set at time"

        if eval_online:
            assert eval_func is not None, "eval func must be assigned"

        if save_results_dir is not None:
            self.save_results_dir = save_results_dir

        if True:
            logger.info(f"Begin eval online, wait...")
            return self._do_eval(data_list, eval_func=eval_func)
        else:
            raise NotImplementedError

    def _do_eval(self, dets, metric: str = 'mAP', use_07_metric: bool = False, ign_diff=True, scale_ranges=None, eval_func=None):
        assert metric in ['mAP', 'recall'], f"Don't support type {metric}"
        assert eval_func is not None, "eval func can't be None"
        # id_mapper = {ann['id']: i for i, ann in enumerate(infos)}
        id_mapper = {file[:-4]: i for i, file in enumerate(self.label_files)}
        det_results, gt_dst = [], []
        for det in dets:
            det_id = det['id']
            det_bboxes = np.concatenate((det['bboxes'], det['scores'][..., None]), axis=-1)
            det_labels = det['labels']
            # det_results.append([det_bboxes[det_labels == i] for i in range(len(self.CLASSES))])
            det_results.append([det_bboxes[det_labels == i] for i in range(1)])
            # logger.info('det_results: {}'.format(len(det_results[0])))

            ann = self.annotations[id_mapper[det_id]][0]
            gt_bboxes = ann[..., :8]
            gt_labels = ann[..., 8]

            # TODO: support Task2
            gt_ann = {'bboxes': gt_bboxes, 'labels': gt_labels}
            gt_dst.append(gt_ann)

        if metric == 'mAP':
            IouTh = np.linspace(0.5, 0.51, int(np.round((0.51 - 0.5) / 0.05)) + 1, endpoint=True)
            # IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
            mAPs = []
            for iou in IouTh:
                mAP = eval_func(det_results, gt_dst, scale_ranges, iou, use_07_metric=use_07_metric, nproc=4)[0]
                mAPs.append(mAP)
            return mAPs, mAPs[0]
        else:
            raise NotImplementedError
