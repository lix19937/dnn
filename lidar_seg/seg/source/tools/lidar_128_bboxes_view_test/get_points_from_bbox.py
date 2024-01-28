import pickle
from pathlib import Path
import numpy
import torch
from det3d.core import box_np_ops
#from det3d.datasets.dataset_factory import get_dataset
from det3d.torchie import Config
from joblib import Parallel, delayed

def get_points_from_bbox(points, gt_boxes):
    offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    gt_boxes = numpy.array(gt_boxes).reshape((1,7))
    point_indices_from_num = box_np_ops.points_in_rbbox(points, gt_boxes)
    gt_points = points[point_indices_from_num[:, -1]]
    gt_points[:, :3] -= gt_points[:, :3]
    return gt_points