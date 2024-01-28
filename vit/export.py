# /**************************************************************
#  * @Copyright: 2021-2022 Copyright 
#  * @Author: lix
#  * @Date: 2022-08-14 18:55:58
#  * @Last Modified by: lix
#  * @Last Modified time: 2023-05-08 10:56:40
#  **************************************************************/

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
from models.core import VisionTransformer, CONFIGS, Block, Mlp
import numpy as np
from loguru import logger

def helper_onnx(model_file):
  import onnx, onnxsim
  shapes_onnx_filename = model_file + "with_sim.onnx"
  model = onnx.load(model_file)
  model_simp, check = onnxsim.simplify(model)
  assert check, "Simplified ONNX model could not be validated"
  onnx.save(model_simp, shapes_onnx_filename)

# 
# torch/onnx/symbolic_helper.py:325: UserWarning: Type cannot be inferred, which might cause exported graph to produce incorrect results.
#   warnings.warn("Type cannot be inferred, which might cause exported graph to produce incorrect results.")
def vit_export():
  logger.info("====================== vit_export ... ======================")
  model_type = "ViT-B_16"
  config = CONFIGS[model_type]

  num_classes = 8 
  img_size = 224
  model = VisionTransformer(config, img_size, zero_head=False, num_classes=num_classes, is_export=True).cuda().eval()

  dummy_input1 = torch.randn((1, 3, 224, 224), device='cuda:0') # 'cuda:0'
  dummy_input2 = torch.tensor([7, 0, 1, 0, 8, 2, 6, 1, 3, 1, 7, 9, 6, 2], device='cuda:0')

  onnx_path = "./output/vit.onnx"
  try:
      torch.onnx.export(model, 
        (dummy_input1, dummy_input2), 
        onnx_path, 
        verbose=False, 
        opset_version=11, 
        enable_onnx_checker=True, 
        do_constant_folding=True)

  except ValueError:
      logger.info("Failed to export to ONNX")
      return 

  logger.info("====================== export done.======================")
  helper_onnx(onnx_path)
  logger.debug("====================== done.======================")


def block_export():
  dummy_input1 = torch.randn(1, 197, 768).cuda()
  model_type = "ViT-B_16"
  config = CONFIGS[model_type]

  model = Block(config, False, True).cuda().eval()
  model(dummy_input1)
  logger.info("====================== export ... ======================")

  onnx_path = "./output/block.onnx"
  torch.onnx.export(model, 
        (dummy_input1,), 
        onnx_path, 
        verbose=False, 
        opset_version=11, 
        enable_onnx_checker=True, 
        do_constant_folding=True)

  helper_onnx(onnx_path)
  logger.debug("====================== done.======================")


def mlp_export():
  dummy_input1 = torch.randn(512, 1, 768).cuda()
  model_type = "ViT-B_16"
  config = CONFIGS[model_type]

  model = Mlp(config, True).cuda().eval()
  model(dummy_input1)
  logger.info("====================== export ... ======================")

  onnx_path = "./output/Mlp.onnx"
  torch.onnx.export(model, 
        (dummy_input1,), 
        onnx_path, 
        verbose=False, 
        opset_version=11, 
        enable_onnx_checker=True, 
        do_constant_folding=True)

  helper_onnx(onnx_path)
  logger.debug("====================== done.======================")


def ln_export():
  dummy_input1 = torch.randn(1, 197, 768).cuda()
  model_type = "ViT-B_16"
  config = CONFIGS[model_type]

  model = torch.nn.LayerNorm(config.hidden_size, eps=1e-6).cuda().eval()
  model(dummy_input1)
  logger.info("====================== export ... ======================")

  onnx_path = "./output/ln.onnx"
  torch.onnx.export(model, 
        (dummy_input1,), 
        onnx_path, 
        verbose=False, 
        opset_version=11, 
        enable_onnx_checker=True, 
        do_constant_folding=True)

  helper_onnx(onnx_path)
  logger.debug("====================== done.======================")


if __name__ == "__main__":
  import os
  out_path = './output'
  if not os.path.exists(out_path):
    os.makedirs(out_path)  

  vit_export()
  mlp_export()
  ln_export()
  block_export()
