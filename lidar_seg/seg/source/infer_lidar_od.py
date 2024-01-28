from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
import time
import os, sys
from models.model import create_model, load_model
from datasets.evaluate.lidar_seg_eval import evaluate
from datasets.evaluate.lidar_eval import lidar_eval, calc_hist
from datasets.dataset_factory import get_dataset
from opts import opts
from loguru import logger

import numpy as np
from tqdm import tqdm
import queue

import torch
import torch.utils.data
import signal

sys.path.append('quantization')
from quantization.quantize_lx import *

sys.path.append('./')
sys.path.append('lib')
from lib.datasets.dataset_factory import get_dataset
from lib.models.model import create_model_quan, load_model
from lib.opts import opts

# 自定义信号处理函数
def my_handler(signum, frame):
    global stop
    stop = True
 
# 设置相应信号处理的handler
signal.signal(signal.SIGINT, my_handler)    #读取Ctrl+c信号

def myexport_onnx(model, file, size=(5, 192, 1024), dynamic_batch=False):
  device = next(model.parameters()).device
  model.float()
  dummy = torch.zeros(1, size[0], size[1], size[2], device=device)
  export_onnx(model, dummy, file, opset_version=13, 
      input_names=["images"], 
      output_names=["output"], 
      dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
  )

def make_model(opt):
    _, quant_nn = initialize()
    model = create_model_quan(opt, phase='val', quant_nn=quant_nn) ########
    model.LONLP = ['resBlock4.pool', 'resBlock3.pool', 'resBlock2.pool', 'resBlock1.pool']
    apply_custom_rules_to_quantizer(model, myexport_onnx, dims=(5, 192, 1024), lonlp = model.LONLP) ### 
     
    model = load_model(model, opt.load_model, user_spec=opt.user_spec)
    model.eval()
    model = model.to('cuda')
    return model

def main(opt):
    Dataset = get_dataset('mini', 'rv_seg')
    opt = opts().parse()
    val_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'val'), #######
        batch_size=1,
        shuffle=False,
        num_workers=112, ##############
        pin_memory=True
    )
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    if not opt.qdq: 
      print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n")   
      model = create_model(opt, 'test')
      model = load_model(model, opt.load_model)
      model = model.to(opt.device)
      model.eval()
    else:
      model = make_model(opt)

    global stop
    stop = False

    total_hist = [np.zeros((opt.ignore_index, opt.ignore_index), dtype=np.int64) for _ in range(4)]
    worst_status = queue.PriorityQueue()
    for batch in tqdm(val_loader):
        if stop:
            break
        with torch.no_grad():
            output = model(batch['input'].to(opt.device))
            output = output[-1]
        
        result = dict()
        result['raw_points'] = batch['input'].detach().numpy()[0]
        result['predict_labels'] = output['seg'].detach().cpu().numpy().reshape(-1)
        result['gt_label'] = batch['gt_segment_label'].detach().numpy()[0].reshape(-1)
        result['path'] = batch['meta'][0]
        # results.append(result)

        inds_1 = (result['raw_points'][0, :] <= 30) & (result['raw_points'][3, :] > 0.1)
        inds_2 = (result['raw_points'][0, :] > 30) & (result['raw_points'][0, :] <= 50) & (result['raw_points'][3, :] > 0)
        inds_3 = (result['raw_points'][0, :] > 50) & (result['raw_points'][3, :] > 0)

        calc_hist(result, (inds_1, inds_2, inds_3), opt.ignore_index, total_hist, worst_status)
    
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    root_dir = os.path.join(opt.save_dir, f'eval_{time_str}')
    evaluate(opt, worst_status, total_hist, Dataset.CLASSES, Dataset.PALETTE, root_dir)


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)


"""python infer_lidar_od.py lidar_od --arch hourglass --down_ratio 4 --gpus 0 --load_model xx.pth"""
