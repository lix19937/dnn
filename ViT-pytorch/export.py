from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torchvision.models as models
import torch
import torch.nn as nn
from models.modeling import VisionTransformer, CONFIGS, Block, Mlp
import numpy as np
import onnxsim 

from loguru import logger

def load_model(model, model_path):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  key = 'epoch'
  if key in checkpoint:
    logger.info('loaded {}, epoch {}'.format(model_path, checkpoint[key]))
  
  key = 'state_dict'
  if key in checkpoint:
      logger.info('found {}'.format(key))
      state_dict_ = checkpoint['state_dict']
  else:
    state_dict_ = checkpoint
  logger.info('state_dict has key num len {}'.format(len(state_dict_)))

  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      logger.info('key:{}'.format(k))
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
   
  model_state_dict = model.state_dict()
  model.load_state_dict(state_dict, strict=False)
  return model


def save_model(path, epoch, model, optimizer=None):
  if isinstance(model, torch.nn.DataParallel):
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  data = {'epoch': epoch,
          'state_dict': state_dict}
  if not (optimizer is None):
    data['optimizer'] = optimizer.state_dict()
  torch.save(data, path)


def helper_onnx(model_file):
  import onnx, onnxsim
  shapes_onnx_filename = model_file + "with_sim.onnx"
  model = onnx.load(model_file)
  model_simp, check = onnxsim.simplify(model)
  assert check, "Simplified ONNX model could not be validated"
  onnx.save(model_simp, shapes_onnx_filename)
  #onnx.save(onnx.shape_inference.infer_shapes(model_simp), shapes_onnx_filename)

def helper_initializer(model_file):
  import onnx
  from onnx import numpy_helper
  model = onnx.load(model_file)

  logger.info("len:{}".format(len(model.graph.initializer)))

  for t in model.graph.initializer:
    w = numpy_helper.to_array(t)
    logger.info("{},{}".format(t.name, w.shape))

def pt2onnx(pth_file_when_export, onnx_file_after_export):
  logger.info("====================== pt2onnx ... ======================")
  model_type = "ViT-B_16"
  # model_type = "R50-ViT-B_16"

  config = CONFIGS[model_type]

  num_classes = 10 
  img_size = 224

  #model = VisionTransformer(config, img_size, zero_head=True, num_classes=num_classes, is_export=True)
  model = VisionTransformer(config, img_size, zero_head=False, num_classes=num_classes, is_export=True)

  # Load the model state dict
  #ckpt = torch.load(pth_file_when_export, map_location="cpu")
  model = load_model(model, pth_file_when_export)
  model = model.to('cpu')
  model.eval()

  dummy_input1 = torch.randn((1, 3, 224, 224), device='cpu') # 'cuda:0'
  dummy_input2 = torch.tensor([7, 0, 1, 0, 8, 2, 6, 1, 3, 1, 7, 9, 6, 2, 0, 8], device='cpu')
  #dummy_input2 = torch.tensor([1], device='cpu')

  try:
      torch.onnx.export(model, 
        (dummy_input1, dummy_input2), 
        onnx_file_after_export, 
        verbose=False, 
        opset_version=11, 
        enable_onnx_checker=True, 
        do_constant_folding=True)

  except ValueError:
      logger.info("Failed to export to ONNX")
      return 

  logger.info("====================== export done.======================")

  helper_onnx(onnx_file_after_export)
  logger.info("====================== All done.======================")

def block_export():
  dummy_input1 = torch.randn(1,197,768)
  model_type = "ViT-B_16"

  config = CONFIGS[model_type]
  model = Block(config, False, True)
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
  logger.info("====================== All done.======================")

def mlp_export():
  dummy_input1 = torch.randn(900,1,768)
  model_type = "ViT-B_16"

  config = CONFIGS[model_type]
  model = Mlp(config, True)
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

  logger.info("====================== All done.======================")

def ln_export():
  dummy_input1 = torch.randn(1,197,768)
  model_type = "ViT-B_16"

  config = CONFIGS[model_type]
  model = torch.nn.LayerNorm(config.hidden_size, eps=1e-6)
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

  logger.info("====================== All done.======================")


if __name__ == "__main__":
  # ln_export();exit()
  # mlp_export();exit()
  block_export();exit()

  pt2onnx('./output/cifar10-100_500_checkpoint.bin', './output/vit.onnx')
  #helper_initializer('./output/vit.onnx')
  # pt2onnx('./output_r50/cifar10-100_500_checkpoint_5.bin', './output_r50/r50vit.onnx')
