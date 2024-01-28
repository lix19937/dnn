# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2023-03-03 11:09:48
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2023-03-03 11:09:48
#  **************************************************************/

import torch
import numpy as np

import sys
sys.path.append('./')
from quantize_lx import *
from rules import *

import onnx
import random
from loguru import logger

def fix_seed():
    logger.info('onnx version:{}'.format(onnx.__version__))
    logger.info('torch version:{}'.format(torch.__version__))

    identical_seed = 1024
    torch.manual_seed(identical_seed)
    torch.cuda.manual_seed_all(identical_seed)
    np.random.seed(identical_seed)
    random.seed(identical_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def export_onnx_nquant(model, file, size, *args, **kwargs):
    model.eval()
    with torch.no_grad():
        tmp = file.replace(".onnx", "_tmp.onnx")
        device = next(model.parameters()).device
        dummy_inputs = torch.rand(size, device=device)

        torch.onnx.export(model, dummy_inputs, tmp, *args, **kwargs)
        logger.info("export onnx done !")
        from onnxsim import simplify
        import os
        model = onnx.load(tmp)
        os.remove(tmp)

        onnx.checker.check_model(model)
        model_simp, check = simplify(model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.checker.check_model(model_simp)
        onnx.save(model_simp, file)
        logger.info("simplify onnx done !")


def export_onnx_quant(model, file, size, input_names=None, output_names=None, dynamic_axes=None):
    device = next(model.parameters()).device
    dummy_inputs = torch.rand(size, device=device)
    export_onnx(model, dummy_inputs, file, opset_version=13,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes)


class NetOpts(object):
    def __init__(self):
        self.model_name = ''
        self.input_dims = ()
        self.input_names = None # ['']
        self.output_names = None # ['']
        self.dynamic_axes = None # {"input": {0: "batch"}, "output": {0: "batch"}}
        self.nn_module = None # torch.nn.module
        self.lonlp = [] # ['']
        self.ignore_policy = None # str 


def pipeline(opts) -> None:
    model_name = opts.model_name
    model = opts.nn_module
    dims = opts.input_dims

    device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)
    model = model.to(device)

    # Just for PTQ benchmark
    export_onnx_nquant(model, file=model_name + "_nquant.onnx", size=dims,
      input_names=opts.input_names, output_names=opts.output_names, dynamic_axes=opts.dynamic_axes)
    logger.info("export_onnx_nquant done")

    # Auto QDQ insert
    replace_to_quantization_module(model, opts.ignore_policy)
    logger.info("replace_to_quantization_module done")

    # Layers of non learning parameters, but can quantize. User must set the layer name according to the forward data flow !!!
    # Custom rules for binding scale, here we donot use dynamic shape
    apply_custom_rules_to_quantizer(model, export_onnx_quant, dims, lonlp=opts.lonlp)
    logger.info("apply_custom_rules_to_quantizer done")

    # QAT onnx
    export_onnx_quant(model, file=model_name + "_quant.onnx", size=dims)
    logger.info("export_onnx_quant done")

