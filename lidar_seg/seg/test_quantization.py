
from loguru import logger
import torch 

import sys  


for it in sys.path:
  logger.info('path:{}'.format(it))

#sys.path.append("/Data/ljw/seg_train_nfs/seg/pytorch-quantization_v2.1.2")
# t = sys.path[-1]
# sys.path[-1] = sys.path[0]
# sys.path[0] = t

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

logger.info('cuda is_available:{}'.format(torch.cuda.is_available()))

calibrator = 'histogram' # (max/histogram)
logger.info('per_channel_quantization') 
quant_desc_input = QuantDescriptor(calib_method=calibrator)
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)

logger.info('per_tensor_quantization')
calibrator = 'histogram'
quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input) #conv
quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input) #convtrans
quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input) #linear

quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

logger.info('done')



