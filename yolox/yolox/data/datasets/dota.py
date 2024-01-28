import os
import cv2
import numpy as np
from loguru import logger

from .datasets_wrapper import Dataset
from yolox.utils import time_synchronized, poly2obb_np, obb2poly_np
from .dota_classes import DOTA_CLASSES

DEBUG = False


class DOTADataset(Dataset):

    def __init__(self, name="train", data_dir=None, img_size=(480, 480), preproc=None, cache=False,
                 save_results_dir=None):
        super().__init__(img_size)
        self.class_to_ind = dict(zip(DOTA_CLASSES, range(len(DOTA_CLASSES))))

        self.imgs = None
        self.name = name
        self.data_dir = data_dir
        self.img_size = img_size
        self.imgs_dir = os.path.join(data_dir, name, 'image')
        self.labels_dir = os.path.join(data_dir, name, 'label')
        self.label_files = os.listdir(self.labels_dir)
        self.annotations = [self.load_anno_along_ids(index) for index in range(len(self.label_files))]
        self.preproc = preproc
        self.save_results_dir = save_results_dir
        if cache:
            self._cache_images()

    def __len__(self):
        return len(self.label_files)

    def load_anno_along_ids(self, index):
        """
        Return:
        res: the label information of per image, [n_gt_per_img, 9]
        9 contain [8 * polys of dets, labels]
        """

        file_name = self.label_files[index]
        res = np.loadtxt(os.path.join(self.labels_dir, file_name), dtype=str)
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
        resized_img = cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_LINEAR, ).astype(np.float32)
        return resized_img

    def load_image(self, index):
        file_name = self.annotations[index][3]
        img_file = os.path.join(self.imgs_dir, file_name[:-3] + 'png')
        img = cv2.imread(img_file)
        if DEBUG:
            print(img_file)
            cv2.imshow('ori', img)

        assert img is not None
        return img

    def pull_item(self, index):
        res, img_info, resized_info, file_name = self.annotations[index]
        img = self.load_resized_img(index)

        return img, res.copy(), img_info, file_name[:-4]

    @Dataset.mosaic_getitem
    def __getitem__(self, index):
        """
        One image / label pair for the given index is picked up and pre-processed.

        Args:
            index (int): data index

        Returns:
            img (numpy.ndarray): pre-processed image
            padded_labels (torch.Tensor): pre-processed label data.
                The shape is :math:`[max_labels, 5]`.
                each label consists of [class, xc, yc, w, h, angle]:
                    class (float): class index.
                    xc, yc (float) : center of bbox whose values range from 0 to 1.
                    w, h (float) : size of bbox whose values range from 0 to 1.
            info_img : tuple of h, w.
                h, w (int): original shape of the image
            img_id (int): the name of image(like P2021.png, and it is img id will be P2021).
        """
        img, target, img_info, img_id = self.pull_item(index)
        new_target = []
        for t in target:
            new_target.append(poly2obb_np(t))
        if len(new_target) == 0:
            new_target = np.empty((0, 6))
        else:
            new_target = np.concatenate(new_target, axis=0)
        if self.preproc is not None:
            img, new_target = self.preproc(img, new_target, self.input_dim)

        if DEBUG:
            print(new_target.shape)
            # label = target[..., :-1].reshape(-1, 4, 1, 2)
            label = obb2poly_np(new_target[..., 1:6])
            label = label.reshape(-1, 4, 1, 2)
            print('label: ', label)
            frame = img.transpose(1, 2, 0)
            frame = cv2.drawContours(cv2.UMat(frame).get(), label.astype(np.int), -1, (0, 255, 0), 2)
            cv2.imshow('frame', frame)
            if cv2.waitKey(0) == ord('q'):
                exit(0)

        return img, new_target, img_info, img_id

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
