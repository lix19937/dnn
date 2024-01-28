from __future__ import absolute_import, division, print_function

import _init_paths
import sys
import os
import os.path as osp
sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

import torch
import torch.utils.data
from opts import opts
from models.model import create_model, load_model
from datasets.dataset_factory import get_dataset
from tqdm import tqdm
import numpy as np

def inference(opt):
    Dataset = get_dataset('new_lidar', 'rv_seg')
    infer_loader = torch.utils.data.DataLoader(
        Dataset(opt, 'test'),
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False
    )
    if opt.gpus[0] >= 0:
        opt.device = torch.device('cuda')
    else:
        opt.device = torch.device('cpu')
    print('Creating model...')
    model = create_model(opt, 'test')
    model = load_model(model, opt.load_model)
    model = model.to(opt.device)
    model.eval()
    global stop
    stop = False
    for batch in tqdm(infer_loader):
        with torch.no_grad():
            output = model(batch['input'].to(opt.device), batch['indices'].to(opt.device),
            batch['mask'].to(opt.device), batch['aligned_cloud'].to(opt.device))
            output = output[-1]
        
        path = batch['meta']['path'][0]
        pc_size = int(batch['meta']['pc_size'][0])
        label = output['seg'].detach().cpu().numpy().reshape(-1).astype(np.uint8)[:pc_size]
        # label[label > 2] += 1
        # label[label > 18] += 1
        label.tofile(path.replace('lidar', 'label_bin')[:-4] + '.bin')


if __name__ == '__main__':
    opt = opts().parse()
    inference(opt)
