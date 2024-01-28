
from asyncio.log import logger
from random import random
from polygraphy.logger import G_LOGGER

G_LOGGER.verbosity(G_LOGGER.ULTRA_VERBOSE)

import numpy as np
import tensorrt as trt
from polygraphy.backend.trt import EngineFromBytes, TrtRunner 
from polygraphy.backend.common import BytesFromPath
import os
import random

from loguru import logger as MyLOG
import json

INPUT_NAME = "images"
INPUT_SHAPE = (1,3,640,640)


def save_byrow(x, file_name, fmt = "%.6f", delimiter=" "):  
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
    np.savetxt(file_name + shape_flg, x, fmt=fmt, delimiter=delimiter)   
  if leng > 2:
    cs = 1
    for i in range(leng - 2):
      cs = cs*shape[i]

    new_shape = (cs, shape[-2], shape[-1])
    rx = x.reshape(new_shape)
    with open(file_name + shape_flg, 'w') as f:
      for i in range(new_shape[0]):
        np.savetxt(f, rx[i], fmt=fmt, delimiter=delimiter)

def loadtxt(file_name, shape):
  a = np.loadtxt(file_name, dtype=float)
  a = a.reshape(shape).astype(np.float32)
  return a

def build_engine(onnx_file_path, engine_file_path, layerinfo_json):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")

    config = builder.create_builder_config()
    builder.max_batch_size = 1 
    print('>>> num_layers:', network.num_layers) 

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ",
                engine_file_path)

    print("Creating Tensorrt Engine") 

    # config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))
    config.max_workspace_size = 2 << 30
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_flag(trt.BuilderFlag.INT8)
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    engine = builder.build_engine(network, config)
    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())
    print("Serialized Engine Saved at: ", engine_file_path)    
  
    inspector = engine.create_engine_inspector()
    layer_json = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    with open(layerinfo_json, "w") as fj:
      json.dump(layer_json, fj)
    return engine

def infer(plan_name):
    MyLOG.info("EngineFromBytes ...")
    engine = EngineFromBytes(BytesFromPath(plan_name))
    with TrtRunner(engine) as runner:
          feed_dict = {INPUT_NAME: np.random.random_sample(INPUT_SHAPE).astype(np.float32)}

          outputs = runner.infer(feed_dict)

          #save_byrow(outputs["output"], "Out.outputs_scores")

          MyLOG.info("Inference succeeded")
    MyLOG.info("Inference done!")


if __name__ == "__main__":  
    MyLOG.info("{}".format(trt.__version__, trt.__file__))

    onnx_name = "./fp321112_275_3head.onnx" # qat1112_3head

    plan_name = onnx_name + "._3090.plan"
    layerinfo_json = onnx_name +"._3090.json"

    build_engine(onnx_name, plan_name, layerinfo_json)

    infer(plan_name)
