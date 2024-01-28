
import numpy as np

import sys
import torch 
from loguru import logger

sys.path.append('quantization')
from quantization.quantize_lx import *

import torch
sys.path.append('./')
sys.path.append('lib')

from lib.models.model import create_model_quan
from lib.opts import opts

seed=317
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

calibrator = ["max", "histogram"][1]
percentileList = [99.9, 99.99, 99.999, 99.9999]

#  python3 ./nv_calib.py lidarv4  --qdq   --down_ratio 4 --gpus 4

def myexport_onnx(model, file, size=(5, 192, 1024), dynamic_batch=False):
  device = next(model.parameters()).device
  model.float()
  dummy = torch.zeros(1, size[0], size[1], size[2], device=device)
  export_onnx(model, dummy, file, 
      opset_version=13, 
      input_names=["input"],
      output_names=["output1", "output2"], 
      dynamic_axes={"input": {0: "batch"}, "output1": {0: "batch"}, "output2": {0: "batch"}} if dynamic_batch else None
  )

def make_model(opt, ckpt, quant_nn, phase='train'):
    model = create_model_quan(opt, phase=phase, quant_nn=quant_nn) ########

    model.LONLP = ['resBlock4.pool', 'resBlock3.pool', 'resBlock2.pool', 'resBlock1.pool']
    apply_custom_rules_to_quantizer(model, myexport_onnx, dims=(5, 192, 1024), lonlp = model.LONLP, local_rank=opt.local_rank) ### s2 
    
    epoch = 0
    checkpoint = torch.load(ckpt, map_location=lambda storage, loc: storage)
    state_dict_ = checkpoint['state_dict']
    state_dict = {}

    # convert data_parallal to model
    for k in state_dict_:
        if k.startswith('module') and not k.startswith('module_list'):
            state_dict[k[7:]] = state_dict_[k]
        else:
            state_dict[k] = state_dict_[k]
    model_state_dict = model.state_dict()

    # check loaded parameters and created model parameters
    msg = 'If you see this, your model does not fully load the ' + \
          'pre-trained weight. Please make sure ' + \
          'you have correctly specified --arch xxx ' + \
          'or set the correct --num_classes for your own dataset.'
    for k in state_dict:
        if k in model_state_dict:
            if state_dict[k].shape != model_state_dict[k].shape:
                print('Skip loading parameter {}, required shape{}, '
                      'loaded shape{}. {}'.format(
                          k, model_state_dict[k].shape, state_dict[k].shape, msg))
                state_dict[k] = model_state_dict[k]
        else:
            print('Drop parameter {}.'.format(k) + msg)
    for k in model_state_dict:
        if not (k in state_dict):
            print('No param {}.'.format(k) + msg)
            state_dict[k] = model_state_dict[k]
    model.load_state_dict(state_dict)

    if 'epoch' in checkpoint:
        epoch = checkpoint['epoch']
        logger.info("checkpoint epoch:{}".format(epoch))

    if 'state_dict' in checkpoint:
        logger.info("checkpoint conv1.bias:{}".format(state_dict['downCntx.conv1.bias'].view(1,-1)))
        logger.info("checkpoint conv2.bias:{}".format(state_dict['downCntx.conv2.bias'].view(1,-1)))
        if 'downCntx.conv1._input_quantizer._amax' in state_dict:
          logger.info("checkpoint conv1._input_quantizer._amax:{}".format(state_dict['downCntx.conv1._input_quantizer._amax'].view(1,-1)))
          logger.info("checkpoint conv1._weight_quantizer._amax:{}".format(state_dict['downCntx.conv1._weight_quantizer._amax'].view(1,-1)))

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
    model.eval()
    return model

## python3 ./nv_calib.py lidarv4  --qdq   --down_ratio 4 --gpus 4


def run():
  logger.info(torch.__version__) # A100 1.10.2+cu111
  logger.info('Parse...')
  opt = opts().parse()

  ptq_pth_file = opt.ptq_pth_file  
  ptq_onnx = ptq_pth_file.replace(".pth", '_reload.onnx') 

  logger.info('Init quan {}...'.format(opt.local_rank))
  _, quant_nn = initialize()

  logger.info('Build QDQ model {}...'.format(opt.local_rank))
  model = make_model(opt=opt, ckpt=ptq_pth_file, quant_nn=quant_nn, phase='val')
  myexport_onnx(model, ptq_onnx)
  logger.info('Done')

 
if __name__ == "__main__":  
    run()