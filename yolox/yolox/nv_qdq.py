

import sys  

#egg_path='/home/igs/fusion_se_sod_training-QAT/pytorch-quantization'
egg_path='/Data/tet/fusion_se_sod_finetune/pytorch-quantization'
sys.path.append(egg_path)

t = sys.path[-1]
sys.path[-1] = sys.path[0]
sys.path[0] = t

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules

class NVQDQ():
  def __init__(self):
    calibrator = ["max", "histogram"][1]

    self.quant_nn = quant_nn
    self.calib = calib

    # for input 
    quant_desc_input = QuantDescriptor(calib_method=calibrator, axis=None)
    self.quant_nn.QuantConv2d.set_default_quant_desc_input(quant_desc_input)
    # self.quant_nn.QuantConvTranspose2d.set_default_quant_desc_input(quant_desc_input)
    self.quant_nn.QuantLinear.set_default_quant_desc_input(quant_desc_input)
    self.quant_nn.QuantConv2d_WeightOnly.set_default_quant_desc_input(quant_desc_input)

    # for weight
    # quant_desc_weight = QuantDescriptor(calib_method=calibrator, axis=None)
    # self.quant_nn.QuantConv2d.set_default_quant_desc_weight(quant_desc_weight)
    # self.quant_nn.QuantConv2d_WeightOnly.set_default_quant_desc_weight(quant_desc_weight)
    # self.quant_nn.QuantConvTranspose2d.set_default_quant_desc_weight(quant_desc_weight)
    # self.quant_nn.QuantLinear.set_default_quant_desc_weight(quant_desc_weight)

QDQ = NVQDQ()   
