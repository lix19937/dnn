from __future__ import absolute_import, division, print_function

import _init_paths

import os

import torch
import torch.utils.data
import torch.distributed as dist
from opts import opts
from models.model import create_model, load_model, save_model
from logger import Logger
from datasets.dataset_factory import get_dataset
from trains.train_factory import train_factory
from trains.lr_scheduler import update_lr, finetune_update_lr
from nv_calib import run
import datetime
import numpy as np

def main(opt):
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = not opt.not_cuda_benchmark and not opt.test
    opt = opts().parse()
    # if do not use distributed module, set single-node environment
    if 'MASTER_ADDR' not in os.environ:
        print('train model in dp')
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=10))# 

    is_master = (opt.local_rank == 0)#############
    if is_master:
        logger = Logger(opt)
        print(opt)

    # os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
    torch.cuda.set_device(opt.local_rank)
    opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')

    print('Creating model...')
    if not opt.qdq:
        model = create_model(opt, 'train')
        optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
        start_epoch = 0
        if opt.load_model != '':
            model, optimizer, start_epoch = load_model(
                model, opt.load_model, optimizer, opt.resume, opt.lr)
    else:
        model, optimizer, start_epoch = run(opt)
        # from ckpt in some epoch 
        if opt.resume:
            start_lr = update_lr(start_epoch, 10, 1e-5, opt.num_epochs, opt.lr, 0.8)
            for param_group in optimizer.param_groups:
                param_group['lr'] = start_lr
            print('Resumed optimizer with start lr:{}  @ start_epoch:{}'.format(start_lr, start_epoch))
        else:
            start_epoch = 0
            start_lr = finetune_update_lr(start_epoch, opt.lr)

    print('Setting up data...')
    DatasetSeg = get_dataset('mini', 'rv_seg')
    train_dataset = DatasetSeg(opt, 'train')
    val_loader = torch.utils.data.DataLoader(
        DatasetSeg(opt, 'val'),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True
    )
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=opt.local_rank, shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True
    )

    Trainer = train_factory['lidar_seg']
    trainer = Trainer(opt, model, opt.local_rank, optimizer)
    trainer.set_device(opt.device)

    print('Starting training...save_dir:{} start_epoch:{}, num_epochs:{}'.format(opt.save_dir, start_epoch+1, opt.num_epochs))
    best = 1e10
    cur_lr = start_lr
    for epoch in range(start_epoch + 1, opt.num_epochs + 1):
        train_sampler.set_epoch(epoch)  # Sets the epoch as random seed.
        mark = epoch if opt.save_all else 'last'
        log_dict_train, _ = trainer.train(epoch, train_loader)##=========================
        if is_master:
            logger.write('epoch: {} |'.format(epoch))
            for k, v in log_dict_train.items():
                logger.scalar_summary('train_{}'.format(k), v, epoch)
                logger.write('{} {:8f} | '.format(k, v))
        
            if opt.val_intervals > 0 and epoch > 120 and epoch % opt.val_intervals == 0:
                save_model(os.path.join(opt.save_dir, 'model_{}.pth'.format(mark)), epoch, model, optimizer)
                with torch.no_grad():
                    log_dict_val, preds = trainer.val(epoch, val_loader)##=========================
                for k, v in log_dict_val.items():
                    logger.scalar_summary('val_{}'.format(k), v, epoch)
                    logger.write('{} {:8f} | '.format(k, v))
                if log_dict_val[opt.metric] < best:
                    best = log_dict_val[opt.metric]
                    save_model(os.path.join(opt.save_dir, 'model_best.pth'), epoch, model)
            else:
                save_model(os.path.join(opt.save_dir, 'model_last.pth'), epoch, model, optimizer)
            if is_master:
                logger.write('\n')

        lr = update_lr(epoch, 10, 1e-4, opt.num_epochs, opt.lr, 0.8)
        # lr = finetune_update_lr(epoch, cur_lr)
        if is_master:
            print('epoch {}, Update LR to {}, from IR {}'.format(epoch, lr, cur_lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        cur_lr = lr
    
    if is_master:
        logger.close()


if __name__ == '__main__':
    opt = opts().parse()
    main(opt)
