# encoding: utf-8
import os
from loguru import logger

import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler

from yolox.exp import Exp as MyExp
from yolox.data import get_yolox_datadir, get_data_type
from yolox.data import SAICDataset


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.head_type = 'saic'
        self.num_classes = 1
        self.depth = 0.33
        self.width = 0.50
        self.warmup_epochs = 1

        #
        self.data_dir = get_yolox_datadir()
        self.data_type = get_data_type()
        self.mean_std_dict = {'snr': dict(mean=[127, 107, 157], std=[56.5115, 29.692, 46.7419]),
                              'height': dict(mean=[127, 107, 153], std=[56.5115, 29.692, 49.4464])}
        self.mean_std = self.mean_std_dict.get(self.data_type)

        # ---------- transform config ------------ #
        self.mosaic_prob = 0.0
        self.mixup_prob = 0.0
        self.hsv_prob = 0.0
        self.flip_prob = 0.5

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import SAICDataset, TrainTransform, DataLoader, InfiniteSampler, worker_init_reset_seed
        from yolox.utils import wait_for_the_master, get_local_rank
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = SAICDataset(data_dir=get_yolox_datadir(), img_size=self.input_size, preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, mean_std=self.mean_std))

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)
        batch_sampler = BatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "batch_sampler": batch_sampler, "worker_init_fn": worker_init_reset_seed}

        # Make sure each process has different random seed, especially for 'fork' method
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import ValTransform

        valdataset = SAICDataset(data_dir=get_yolox_datadir(), name='val', img_size=self.test_size, preproc=ValTransform(), )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True, "sampler": sampler, "batch_size": batch_size}
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = VOCEvaluator(dataloader=val_loader, img_size=self.test_size, confthre=self.test_conf, nmsthre=self.nmsthre, num_classes=self.num_classes, )
        return evaluator
