from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import numpy as np


class opts(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        # basic experiment setting
        self.parser.add_argument('task', default='lidar',
                                 help='should be lidar')
        self.parser.add_argument('--local_rank', default=0, type=int,
                                 help='node rank for distributed training')
        self.parser.add_argument('--dataset', default='lidar128',
                                 help='lidar128 | luminar')
        self.parser.add_argument('--test', action='store_true', default=False)
        self.parser.add_argument('--load_model', default='',
                                 help='path to pretrained model')
        self.parser.add_argument('--resume', action='store_true',
                                 help='resume an experiment. '
                                      'Reloaded the optimizer parameter and '
                                      'set load_model to model_last.pth '
                                      'in the exp dir if load_model is empty.')

        # system
        self.parser.add_argument('--gpus', default='0',
                                 help='-1 for CPU, use comma for multiple gpus')
        self.parser.add_argument('--num_workers', type=int, default=4,
                                 help='dataloader threads. 0 for single-thread.')
        self.parser.add_argument('--not_cuda_benchmark', action='store_true',
                                 help='disable when the input size is not fixed.')
        self.parser.add_argument('--seed', type=int, default=317,
                                 help='random seed')  # from CornerNet

        # log
        self.parser.add_argument('--print_iter', type=int, default=0,
                                 help='disable progress bar and print to screen.')
        self.parser.add_argument('--hide_data_time', action='store_true',
                                 help='not display time during training.')
        self.parser.add_argument('--save_all', action='store_true',
                                 help='save model to disk every 5 epochs.')
        self.parser.add_argument('--metric', default='loss',
                                 help='main metric to save best model')
        self.parser.add_argument('--vis_thresh', type=float, default=0.3,
                                 help='visualization threshold.')
        self.parser.add_argument('--debugger_theme', default='white',
                                 choices=['white', 'black'])

        # model
        self.parser.add_argument('--arch', default='salsa',
                                 help='model architecture. Currently tested'
                                      'resnet34 | salsa | unet')
        self.parser.add_argument('--down_ratio', type=int, default=4,
                                 help='output stride. Currently only supports 4.')

        # input
        self.parser.add_argument('--input_res', type=int, default=-1,
                                 help='input height and width. -1 for default from '
                                 'dataset. Will be overriden by input_h | input_w')
        self.parser.add_argument('--input_h', type=int, default=-1,
                                 help='input height. -1 for default from dataset.')
        self.parser.add_argument('--input_w', type=int, default=-1,
                                 help='input width. -1 for default from dataset.')
        self.parser.add_argument('--input_c', type=int, default=16,
                                 help='feature map channels for lidar data. 16 for default from dataset.')

        # train
        self.parser.add_argument('--lr', type=float, default=1.25e-4,
                                 help='learning rate for batch size 32.')
        # self.parser.add_argument('--lr_step', type=str, default='90,120',
        #                          help='drop learning rate by 10.')
        self.parser.add_argument('--num_epochs', type=int, default=40000,
                                 help='total training epochs.')
        self.parser.add_argument('--batch_size', type=int, default=32,
                                 help='batch size')
        self.parser.add_argument('--num_iters', type=int, default=-1,
                                 help='default: #samples / batch_size.')
        self.parser.add_argument('--val_intervals', type=int, default=10,
                                 help='number of epochs to run validation.')
        self.parser.add_argument('--trainval', action='store_true',
                                 help='include validation in training and '
                                      'test on test set')
        # lidar od
        self.parser.add_argument('--aug_lidar', type=float, default=0.5,
                                 help='probability of applying lidar data augmentation.')

        # segmentation
        self.parser.add_argument('--ignore_index', type=int, default=10, help='label index which do not train')
        self.parser.add_argument('--align_size', type=int, default=100000, help='point cloud max size')
        self.parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight_decay')
        self.parser.add_argument('--exp_id', type=str, default='default', help='your_test_name')
		
        self.parser.add_argument('--user_spec', default=False, action='store_true', help='use spec way load model')
        self.parser.add_argument('--qdq', default=False, action='store_true', help='use qdq')

        self.parser.add_argument('--onnx_out', default='lidarnet_seg_qat.onnx', help='onnx name')               
        self.parser.add_argument('--fp32_ckpt_file', default='f32.pth', help='pth name')
        self.parser.add_argument('--ptq_pth_file', default='', help='ptq pth file')
        self.parser.add_argument('--calib_dataset_path', default='', help='calib_dataset path')
        self.parser.add_argument('--exec_calib', default=False, action='store_true', help='use calib')


    def parse(self, args=''):
        if args == '':
            opt = self.parser.parse_args()
        else:
            opt = self.parser.parse_args(args)

        opt.gpus_str = opt.gpus
        opt.gpus = [int(gpu) for gpu in opt.gpus.split(',')]
        opt.gpus = [i for i in range(
            len(opt.gpus))] if opt.gpus[0] >= 0 else [-1]

        if opt.trainval:
            opt.val_intervals = 100000000

        opt.root_dir = os.path.join(os.path.dirname(__file__), '..', '..')
        opt.data_dir = os.path.join(opt.root_dir, 'data')
        opt.exp_dir = os.path.join(opt.root_dir, 'exp', opt.task)
        opt.save_dir = os.path.join(opt.exp_dir, opt.exp_id)
        opt.debug_dir = os.path.join(opt.save_dir, 'debug')
        print('The output will be saved to ', opt.save_dir)

        if opt.resume and opt.load_model == '':
            model_path = opt.save_dir[:-4] if opt.save_dir.endswith('TEST') \
                else opt.save_dir
            opt.load_model = os.path.join(model_path, 'model_last.pth')
        return opt

    def init(self, args=''):
        default_dataset_info = {
            'ctdet': {'default_resolution': [512, 512], 'num_classes': 80,
                      'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                      'dataset': 'coco'},
            'exdet': {'default_resolution': [512, 512], 'num_classes': 80,
                      'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                      'dataset': 'coco'},
            'multi_pose': {
                'default_resolution': [512, 512], 'num_classes': 1,
                'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                'dataset': 'coco_hp', 'num_joints': 17,
                'flip_idx': [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10],
                             [11, 12], [13, 14], [15, 16]]},
            'ddd': {'default_resolution': [384, 1280], 'num_classes': 3,
                    'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225],
                    'dataset': 'kitti'},
            'lidar_od': {'default_resolution': [512, 512], 'num_classes': 12,
                         'mean': [0.408, 0.447, 0.470], 'std': [0.289, 0.274, 0.278],
                         'dataset': 'lidar128'}
        }

        class Struct:
            def __init__(self, entries):
                for k, v in entries.items():
                    self.__setattr__(k, v)
        opt = self.parse(args)
        dataset = Struct(default_dataset_info[opt.task])
        opt.dataset = dataset.dataset
        opt = self.set_input_info_and_heads(opt)
        return opt
