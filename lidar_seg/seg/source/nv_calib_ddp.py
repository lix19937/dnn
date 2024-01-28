
import numpy as np

import sys
import torch 
from torch.autograd import Variable
from loguru import logger

sys.path.append('quantization')
from quantization.quantize_lx import *

import torch
sys.path.append('./')
sys.path.append('lib')

from lib.datasets.dataset_factory import get_dataset, get_dataset_for_calib
from lib.models.model import create_model_quan
from lib.opts import opts
from tqdm import tqdm

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import datetime

seed=317
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

calibrator = ["max", "histogram"][1]
percentileList = [99.9, 99.99, 99.999, 99.9999]

def myexport_onnx(model, file, size=(5, 192, 1024), dynamic_batch=False):
  device = next(model.parameters()).device
  model.float()
  dummy = torch.zeros(1, size[0], size[1], size[2], device=device)
  export_onnx(model, dummy, file, 
      opset_version=13, 
      input_names=["input"],
      output_names=["output1", "output2"], 
      dynamic_axes={"input": {0: "batch"}, "output1": {0: "batch"}, "output2": {0: "batch"}} if dynamic_batch else None,
      enable_onnx_checker=False
  )

def make_model(opt, ckpt, quant_nn, phase='train'):
    model = create_model_quan(opt, phase=phase, quant_nn=quant_nn) ########

    model.LONLP = ['resBlock4.pool', 'resBlock3.pool', 'resBlock2.pool', 'resBlock1.pool']
    apply_custom_rules_to_quantizer(model, myexport_onnx, dims=(5, 192, 1024), lonlp = model.LONLP, local_rank=opt.local_rank) ### s2 
    
    epoch = 0
    checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
    if not isinstance(checkpoint, dict):
        logger.info(checkpoint.__dict__.keys())
        model.load_state_dict(checkpoint.state_dict())
    else:
        # logger.info("checkpoint keys:{}".format(checkpoint.keys()))
        if 'epoch' in checkpoint:
            epoch = checkpoint['epoch']
            logger.info("checkpoint epoch:{}".format(epoch))

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            logger.info("checkpoint conv1.bias:{}".format(state_dict['downCntx.conv1.bias'].view(1,-1)))
            logger.info("checkpoint conv2.bias:{}".format(state_dict['downCntx.conv2.bias'].view(1,-1)))
            if 'downCntx.conv1._input_quantizer._amax' in state_dict:
              logger.info("checkpoint conv1._input_quantizer._amax:{}".format(state_dict['downCntx.conv1._input_quantizer._amax'].view(1,-1)))
              logger.info("checkpoint conv1._weight_quantizer._amax:{}".format(state_dict['downCntx.conv1._weight_quantizer._amax'].view(1,-1)))
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint) 

    if opt.local_rank == 0:
        if 'downCntx.conv1._input_quantizer._amax' in model.state_dict():  
            logger.info("epoch:{}\nconv1._input_quantizer\n{}\nconv1._weight_quantizer\n{}\
              \nconv1.bias\n{}\nconv2._input_quantizer\n{}\nconv2._weight_quantizer\n{}\nconv2.bias\n{}".format(epoch, 
              model.state_dict()['downCntx.conv1._input_quantizer._amax'].view(1,-1), 
              model.state_dict()['downCntx.conv1._weight_quantizer._amax'].view(1,-1),
              model.state_dict()['downCntx.conv1.bias'].view(1,-1),
              model.state_dict()['downCntx.conv2._input_quantizer._amax'].view(1,-1),
              model.state_dict()['downCntx.conv2._weight_quantizer._amax'].view(1,-1),
              model.state_dict()['downCntx.conv2.bias'].view(1,-1)))
    model = model.to('cuda')
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), opt.lr, weight_decay=opt.weight_decay)
    if isinstance(checkpoint, dict) and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, epoch

def ddp_save(model, ptq_out_file, is_master:bool):
  if is_master:
      if isinstance(model, torch.nn.DataParallel):
          state_dict = model.module.state_dict()
      else:
          state_dict = model.state_dict()

      data = {'epoch': -1, 'state_dict': state_dict}
      torch.save(data, ptq_out_file)
      #onnx_path = ptq_out_file.replace(".pth", '.onnx')  
      #myexport_onnx(model, onnx_path)


def ptq_calib(model, ptq_out_file, trainLoader, nCalibrationBatch, calib, quant_nn, is_master = True):
  with torch.no_grad():
      logger.info("enable calib ...")
      for _, module in model.named_modules():
          if isinstance(module, quant_nn.TensorQuantizer):
              if module._calibrator is not None:
                  module.disable_quant()
                  module.enable_calib()
              else:
                  module.disable()
      
      logger.info("load calib data ...")
      for i, (xTrain) in tqdm(enumerate(trainLoader), total=nCalibrationBatch):
          if i >= nCalibrationBatch:
              break
          model(Variable(xTrain['input']).cuda())#### forward

      logger.info("calib load data done, disable calib ...")
      for _, module in model.named_modules():
          if isinstance(module, quant_nn.TensorQuantizer):
              logger.info("_calibrator:{}".format(module._calibrator))
              if module._calibrator is not None:
                  module.enable_quant()
                  module.disable_calib()
              else:
                  module.enable()

      def computeArgMax(model, calib, **kwargs):
          for name, module in model.named_modules():
              if isinstance(module, quant_nn.TensorQuantizer) and module._calibrator is not None:
                  if isinstance(module._calibrator, calib.MaxCalibrator):
                      module.load_calib_amax()
                  else:
                      module.load_calib_amax(**kwargs)
                  print(F"{name:40}: {module}")

      logger.info("computeArgMax ...")
      if calibrator == "max":
          computeArgMax(model, calib, method="max")
          modelName = ptq_out_file.replace(".pth", "_max_.pth")
          ddp_save(model, modelName, is_master)
      else:
          logger.info("computeArgMax percentile ...")
          for percentile in percentileList:
              computeArgMax(model, calib, strict=False, method="percentile")  
              modelName = ptq_out_file.replace(".pth", "_p" + str(percentile) + "_.pth")
              ddp_save(model, modelName, is_master)

          logger.info("computeArgMax mse/entropy ...")
          for method in ["mse", "entropy"]:
              computeArgMax(model, calib, strict=False, method=method)
              modelName = ptq_out_file.replace(".pth", "_" + method + "_.pth")
              ddp_save(model, modelName, is_master)

      logger.info("Succeeded calibrating model in pyTorch! {}".format(ptq_out_file))


def run(cfg = None):
  logger.info(torch.__version__) # A100 1.10.2+cu111
  logger.info('Parse...')
  
  if cfg is None:
    opt = opts().parse()
  else:
    opt = cfg 

  ######################################################################
  if 'MASTER_ADDR' not in os.environ:
      print('train model in dp')
      os.environ['RANK'] = '0'
      os.environ['WORLD_SIZE'] = '1'
      os.environ['MASTER_ADDR'] = '127.0.0.1'
      os.environ['MASTER_PORT'] = '29500'
  dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=5)) 
  torch.cuda.set_device(opt.local_rank)
  opt.device = torch.device('cuda' if opt.gpus[0] >= 0 else 'cpu')
  ######################################################################

  logger.info('Init quan {}...'.format(opt.local_rank))
  calib, quant_nn = initialize()
  logger.info('Build QDQ model {}...'.format(opt.local_rank))
  model, optimizer, epoch = make_model(opt=opt, ckpt=opt.fp32_ckpt_file, quant_nn=quant_nn)
  
  ######################################################################
  #model = model.to(opt.device)
  model = DistributedDataParallel(
    model, device_ids=[opt.local_rank], output_device=opt.local_rank,
    find_unused_parameters=True)
  is_master = (opt.local_rank == 0) 
  
  ######################################################################

  if opt.exec_calib:
      logger.info('Setting up data {}...'.format(opt.local_rank))
      # DatasetSeg = get_dataset('mini', 'rv_seg')
      DatasetSeg = get_dataset_for_calib() 
      train_dataset = DatasetSeg(opt, 'train')

      train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=dist.get_world_size(), rank=opt.local_rank, shuffle=True)
      trainLoader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        num_workers=opt.num_workers,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        pin_memory=True,
        drop_last=False
      )

      logger.info('{}'.format(type(trainLoader.__dict__['dataset'].__dict__['images'])))
      logger.info('images total:{}'.format(len(trainLoader.dataset.images)))
      logger.info('{}'.format(trainLoader.__dict__.keys()))

      base_iter = (len(trainLoader.dataset.images) + trainLoader.batch_size -1) //  trainLoader.batch_size
      logger.info('calib iter num {}...'.format(base_iter))
  
      logger.info('PTQ {}...'.format(opt.local_rank))
      if opt.ptq_pth_file is None or opt.ptq_pth_file == '':
          opt.ptq_pth_file = opt.fp32_ckpt_file.replace(".pth", "_calib_" + str(opt.local_rank)+ ".pth")
      ptq_calib(model, ptq_out_file=opt.ptq_pth_file, trainLoader=trainLoader, nCalibrationBatch=base_iter, calib=calib, quant_nn=quant_nn, is_master=is_master) 
      
  logger.info('Done')
  return model, optimizer, epoch


if __name__ == "__main__":  
    run()
