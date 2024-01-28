from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os 
import os.path as osp
import numpy as np


class SegmentationDataset:
    """dataset for semantic segmentation. An example of file structure
    is as followed.

    .. code-block:: none

        ├── data
        │   ├── my_dataset
        │   │   ├── pcd_dir
        │   │   │   ├── xxx{bin}
        │   │   │   ├── yyy{bin}
        │   │   │   ├── zzz{bin}
        │   │   ├── ann_dir
        │   │   │   ├── xxx{bin}
        │   │   │   ├── yyy{bin}
        │   │   │   ├── zzz{bin}

    The img/gt_semantic_seg pair of CustomDataset should be of the same
    except suffix. A valid img/gt_semantic_seg filename pair should be like
    ``xxx{bin}`` and ``xxx{bin}`` (extension is also included
    in the suffix). If filter is given, then ``xxx`` is ignored in txt file.
    Otherwise, all files in ``pcd_dir/``and ``ann_dir`` will be loaded.
    Please refer to ``docs/tutorials/new_dataset.md`` for more details.
    """

    CLASSES = None

    PALETTE = None

    def __init__(self, opt, split):
        workspace_folder = '/'.join(osp.abspath(__file__).split('/')[:-5])
        data_root = osp.join(workspace_folder, 'data', 'seg')
        filter_file = os.path.join(data_root, split + '.txt')
        data_list = os.listdir(data_root)
        filters = []
        if os.path.exists(filter_file):
            filters = open(filter_file, 'r').readlines()
            filters = [_.strip() for _ in filters]
        task_dir = [os.path.join(data_root, _) for _ in data_list if _ in filters]
        self.img_suffix = '.bin'
        # self.ann_dir = [osp.join(_, 'ann_dir', split) for _ in self.data_list if _ not in filters]
        # self.filter_file = osp.join(self.data_root, 'filter.txt')
        self.seg_map_suffix = '.bin'
        self.test_mode = split == 'val'
        self.images = []
        self.labels = []
        self.split = split
        self.opt = opt

        if self.test_mode:
            assert self.CLASSES is not None, \
                '`cls.CLASSES` should be specified when testing'

        # load annotations
        [self.load_annotations(task) for task in task_dir]

        print(f'Loaded {len(self.images)} {self.split} images')

    def __len__(self):
        """Total number of samples of data."""
        return len(self.images)

    def load_annotations(self, task_dir):
        img_dir = os.path.join(task_dir, 'pcd_dir')
        filter_file = os.path.join(task_dir, 'filter.txt')
        ann_dir = os.path.join(task_dir, 'ann_dir')
        if not os.path.exists(img_dir):
            return
        if os.path.exists(filter_file):
            filter_list = open(filter_file, 'r').readlines()
            filter_set = []
            for filter in filter_list:
                a = filter.split('_')[0].strip()
                try:
                    a = float(a)
                    filter_set.append(a)
                except:
                    continue
            last = filter_set[-1]
            filter_set = set(filter_set)
        else:
            last = 1e32
            filter_set = set()

        img_list = sorted(os.listdir(img_dir))
        for img in img_list:
            stamp = float(img.split('_')[0])
            if stamp in filter_set:
                continue
            if stamp > last:
                break
            self.images.append(os.path.join(img_dir, img))
            if ann_dir is not None:
                seg_map = img.replace(self.img_suffix, self.seg_map_suffix)
                self.labels.append(os.path.join(ann_dir, seg_map))
