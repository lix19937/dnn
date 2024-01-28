from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .lidar_seg_trainer import LidarSegTrainer

train_factory = {
    'lidar_seg': LidarSegTrainer
}
