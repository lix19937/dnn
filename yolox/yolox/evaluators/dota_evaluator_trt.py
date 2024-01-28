#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import time
import torch
import itertools
import numpy as np
from loguru import logger
from tqdm import tqdm
from .dota_eval import eval_arb_map

from yolox.utils import gather, is_main_process, obbpostprocess, synchronize, time_synchronized

from polygraphy.backend.trt import EngineFromBytes, TrtRunner 
from polygraphy.backend.common import BytesFromPath

class DOTAEvaluator_TRT:
    """
    DOTA AP Evaluation class.  
    """

    def __init__(self, dataloader, img_size, confthre, nmsthre, num_classes, testdev=False, ign_diff=0):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.ign_diff = ign_diff
        self.best_ap = [0 for _ in range(10)]
        self.best_model = [None for _ in range(10)]
        self.inp_tensor = "images"
        self.outp_tensor ="output"

    def evaluate(self, model, is_half=False, is_distributed=False, decoder=None, trt_file=None, test_size=None):
        """
        DOTA average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by DOTAAPI.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if is_half else torch.cuda.FloatTensor
        model = model.eval()
        if is_half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)
        logger.info("eval trt ...")

        logger.info("EngineFromBytes ...")
        engine = EngineFromBytes(BytesFromPath(trt_file))
        with TrtRunner(engine) as runner:
            #save_byrow(outputs["output"], "Out.outputs_scores")

            logger.info("Inference ...")

            for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
                imgs = imgs.type(tensor_type)

              # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                  start = time.time()

                # imgs = torch.ones(1, 4, test_size[0], test_size[1]).cuda()
                feed_dict = {self.inp_tensor: imgs.cpu().numpy()}
                outputs = runner.infer(feed_dict)[self.outp_tensor]
                outputs = torch.from_numpy(outputs)
                # logger.info("test: {}".format(outputs[0][0]))

                if decoder is not None:
                    # outputs = outputs.unsqueeze(0)
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                # logger.info('outputs: {}'.format(outputs))

                outputs = obbpostprocess(outputs, self.num_classes, self.confthre, self.nmsthre)  # shape(n_pre, 10)
                # logger.info('outputs: {}'.format(outputs))
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end

            data_list.extend(self.convert_to_dota_format(outputs, info_imgs, ids))
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if is_distributed:
            data_list = gather(data_list, dst=0)
            data_list = itertools.chain(*data_list)
            torch.distributed.reduce(statistics, dst=0)

        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_dota_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(outputs, info_imgs[0], info_imgs[1], ids):
            if output is None:
                continue
            output = output.cpu()
            bboxes = output[:, :8]

            # preprocessing: resize
            # scale = min(self.img_size[0] / float(img_h), self.img_size[1] / float(img_w))
            # bboxes /= scale

            cls = output[:, 10]
            # scores = output[:, 8] * output[:, 9]  # todo:change
            scores = output[:, 8]  # todo:change
            pred_data = {"id": img_id,
                         "bboxes": bboxes.numpy(),
                         "labels": cls.numpy(),
                         "scores": scores.numpy()}
            data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_list, statistics, eval_online=False):
        if not is_main_process():
            return 0, 0, None
        logger.info("Evaluate in main process...")

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(["Average {} time: {:.2f} ms".format(k, v) for k, v in zip(["forward", "NMS", "inference"], [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)])])

        info = time_info + "\n"
        logger.info('here 0')

        try:
            mAPs, mAP50 = self.dataloader.dataset.evaluate_detection(data_list, eval_online=eval_online, eval_func=eval_arb_map)

            if mAPs is None or mAP50 is None:
                logger.info('here1')
                return [0], 0, None
            else:
                logger.info('here2')
                logger.info('maps: {},{},{}'.format(mAPs, mAP50, info))
                return mAPs, mAP50, info
        except:
            return [0], 0, None


