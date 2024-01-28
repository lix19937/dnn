
import os
import tensorrt as trt
import json
from loguru import logger as MyLOG

def build_engine(onnx_file, engine_file=None, layerinfo_json=None):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  
    builder = trt.Builder(trt_logger)
    network = builder.create_network(1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read()):
            MyLOG.error('ERROR: Failed to parse the ONNX file {}'.format(onnx_file))
            for error in range(parser.num_errors):
                MyLOG.info(parser.get_error(error))
            return None
    MyLOG.info("Completed parsing ONNX file")

    config = builder.create_builder_config()
    builder.max_batch_size = 1 
    MyLOG.info('num_layers:{}'.format(network.num_layers))

    if engine_file is None:
        engine_file = onnx_file.replace(".onnx", ".plan")
    if os.path.isfile(engine_file):
        try:
            os.remove(engine_file)
        except Exception:
            MyLOG.info("Cannot remove existing file:{}".format(engine_file))

    MyLOG.info("Creating Tensorrt Engine") 

    config.max_workspace_size = 2 << 30
    config.set_flag(trt.BuilderFlag.FP16)
    # config.set_flag(trt.BuilderFlag.INT8)

    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED

    engine = builder.build_engine(network, config)
    with open(engine_file, "wb") as f:
        f.write(engine.serialize())
    MyLOG.info("Serialized Engine Saved at:{}".format(engine_file)) 
 
    if layerinfo_json is not None:
        inspector = engine.create_engine_inspector()
        layer_json = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))

        with open(layerinfo_json, "w") as fj:
          json.dump(layer_json, fj)
    return engine


# if __name__ == "__main__":
    # MyLOG.info("{}".format(trt.__version__, trt.__file__))   
    # eg = build_engine(ONNX_SIM_MODEL_PATH)

