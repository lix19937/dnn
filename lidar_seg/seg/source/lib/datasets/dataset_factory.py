from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.lidar_seg import LidarSeg
from .sample.rv_seg import RvSeg
from .sample.seq_rv_seg import SeqRvSeg
from .sample.lidar_od import LidarOd

from .dataset.new_lidar import NewLidar
from .dataset.luminar import Luminar
from .dataset.luminar_seg import LuminarSeg
from .dataset.mini_data import MiniData

dataset_factory = {
    'new_lidar': NewLidar,
    'luminar': Luminar,
    'segmentation': LuminarSeg,
    'mini': MiniData,
}

_sample_factory = {
    'lidar_od': LidarOd,
    'lidar_seg': LidarSeg,
    'rv_seg': RvSeg,
    'seq_rv_seg': SeqRvSeg
}

def get_dataset(dataset, task):
    class Dataset(dataset_factory[dataset], _sample_factory[task]):
        def __init__(self, opt, split) -> None:
            super().__init__(opt, split)
            _sample_factory[task].__init__(self)
    return Dataset

#   DatasetSeg = get_dataset('mini', 'rv_seg')

def get_dataset_for_calib():
    class Dataset(MiniData, RvSeg):
        def __init__(self, opt, split) -> None:
            super().__init__(opt, split, True)
            RvSeg.__init__(self)
    return Dataset