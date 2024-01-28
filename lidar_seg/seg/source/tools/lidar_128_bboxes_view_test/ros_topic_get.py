#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import rospy
import numpy
import torch
from det3d import torchie
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.models import build_detector
from det3d.torchie.parallel import MegDataParallel, MegDistributedDataParallel
from det3d.torchie.trainer import get_dist_info, load_checkpoint
from det3d.torchie.apis.train import example_convert_to_torch
from det3d.torchie.trainer.trainer import example_to_device
from det3d.utils.dist.dist_common import (all_gather, get_rank, get_world_size, is_main_process, synchronize,)
class PointCloudSubscriber(object):
    def __init__(self) -> None:
        self.sub = rospy.Subscriber("/velodyne_points_syc",
                                     PointCloud2,
                                     self.callback, queue_size=10)
        self.range = [0, -40.0, -3.0, 70.4, 40.0, 1.0]
        self.voxel_size = [0.05, 0.05, 0.1]
        self.max_points_in_voxel = 5
        self.max_voxel_num = 20000
        self.voxel_generator = VoxelGenerator(
            point_cloud_range=self.range,
            voxel_size=self.voxel_size,
            max_num_points=self.max_points_in_voxel,
            max_voxels=self.max_voxel_num,
        )

        self.cfg = torchie.Config.fromfile('/home/wyl/文档/SE-SSD-master/examples/second/configs/config.py')
        if self.cfg.get("cudnn_benchmark", False):  # False
            torch.backends.cudnn.benchmark = True
        self.cfg.data.val.test_mode = True
        self.model = build_detector(self.cfg.model, train_cfg=None, test_cfg=self.cfg.test_cfg)
        self.checkpoint = load_checkpoint(self.model, '/home/wyl/文档/SE-SSD-master/saved_modelexp_se_ssd_v1_8/epoch_60.pth', map_location="cpu")
        if "CLASSES" in self.checkpoint["meta"]:
            self.model.CLASSES = self.checkpoint["meta"]["CLASSES"]
        else:
            pass

        self.model = MegDataParallel(self.model, device_ids=[0])
        self.model = self.model.module
        self.model.eval()
        #self.device = torch.device('cuda')

    def callback(self, msg):
        assert isinstance(msg, PointCloud2)
        # gen=point_cloud2.read_points(msg,field_names=("x","y","z"))
        points = point_cloud2.read_points_list(msg, field_names=("x", "y", "z"))
        point2np = numpy.array(points)
        t = numpy.ones(len(points))
        point2np = numpy.insert(point2np, 3, values=t, axis=1)
        voxels, coordinates, num_points_per_voxel = self.voxel_generator.generate(point2np)
        coordinates = numpy.insert(coordinates, 0, values=numpy.zeros(voxels.shape[0]), axis=1)
        num_voxels = numpy.array([voxels.shape[0]], dtype=numpy.int64)
        res = dict(
            lidar = dict(points = point2np,
                         voxels=dict(
                             voxels=torch.from_numpy(voxels),
                             coordinates=torch.from_numpy(coordinates),
                             num_points=torch.from_numpy(num_points_per_voxel),
                             num_voxels=torch.from_numpy(num_voxels),
                             shape=torch.from_numpy(self.voxel_generator.grid_size),
                         ),
                         ),
        )
        #print(self.model)
        model = self.model
        detections = self.compute_on_dataset(model, res)
        synchronize()
        predictions = self._accumulate_predictions_from_multiple_gpus(detections)

    def compute_on_dataset(model_kitti, device, data, timer=None, show=False):
        '''
            Get predictions by model inference.
                - output: ['box3d_lidar', 'scores', 'label_preds', 'metadata'];
                - detections: type: dict, length: 3769, keys: image_ids, detections[image_id] = output;
        '''

        cpu_device = torch.device("cpu")
        results_dict = {}
        model_kitti = device
        #model_kitti = model_kitti.double()
        device = torch.device('cuda')
        example = data['lidar']['voxels']
        example = example_to_device(example, device=device)
        with torch.no_grad():
            outputs = model_kitti(example, return_loss=False, rescale=not show)  # list_length=batch_size: 8
            print(outputs['box3d_lidar'])
            '''
            for output in outputs:  # output.keys(): ['box3d_lidar', 'scores', 'label_preds', 'metadata']
                token = output["metadata"]["token"]  # token should be the image_id
                for k, v in output.items():
                    if k not in ["metadata", ]:
                        output[k] = v.to(cpu_device)
                results_dict.update({token: output, })
                if i >= 1:
                    prog_bar.update()
            '''
        return results_dict

    def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
        all_predictions = all_gather(predictions_per_gpu)
        if not is_main_process():
            return

        predictions = {}
        for p in all_predictions:
            predictions.update(p)

        return predictions


if __name__ =='__main__':
    rospy.init_node("pointcloud_subscriber")
    PointCloudSubscriber()
    rospy.spin()