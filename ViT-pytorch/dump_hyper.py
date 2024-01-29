# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-03-20 11:09:13
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-03-20 11:09:13
#  **************************************************************/

import onnx
from onnx import numpy_helper
from loguru import logger
import numpy as np
import os

def save_byrow(x, file_name, fmt = "%.6f"):
  shape = x.shape
  leng = len(shape)
  if leng == 1:
    x = x.reshape(1, -1)
    shape = x.shape
    leng = len(shape)
  
  flg = '-'
  b = [str(i) for i in shape] 
  shape_flg = '.'+flg.join(b)

  if leng <= 0:
    return
  if leng == 2:
    np.savetxt(file_name + shape_flg, x, fmt=fmt, delimiter=" ")   
  if leng > 2:
    cs = 1
    for i in range(leng - 2):
      cs = cs*shape[i]

    new_shape = (cs, shape[-2], shape[-1])
    rx = x.reshape(new_shape)
    with open(file_name + shape_flg, 'a') as f:
      for i in range(new_shape[0]):
        np.savetxt(f, rx[i], fmt=fmt, delimiter=" ")

def dump(min, mout):
  model = onnx.load(min)
  save_path = mout
  onnx.checker.check_model(model)

  for i in range(len(model.graph.initializer)): 
    raw_tensor = model.graph.initializer[i]

    raw_np = numpy_helper.to_array(raw_tensor)
    logger.info("{}, {}, {}".format(raw_tensor.name, raw_tensor.data_type, raw_np.shape))
    flg = "%.6f"
    if int(raw_tensor.data_type) == 7:
      flg = "%d"

    ## 
    save_byrow(raw_np, os.path.join(save_path, raw_tensor.name), flg)

if __name__ == "__main__":
  dump("./output/ViT_Attention.onnxwith_sim.onnx", "./output/")
