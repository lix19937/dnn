from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .lidar_od import LidarOdDetector
detector_factory = {
    'lidar_od': LidarOdDetector,
}
