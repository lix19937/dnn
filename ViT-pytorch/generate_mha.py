# QKVToContextVarSeqlenPluginCreator

  # "type_id", nullptr, PluginFieldType::kINT32, 1)); # 0 fp32, 1 fp16, 2 int8
  # "hidden_size", nullptr, PluginFieldType::kINT32, 1));# 512
  # "num_heads", nullptr, PluginFieldType::kINT32, 1));# 8
  # "has_mask", nullptr, PluginFieldType::kINT32, 1));# 0
  # "dq_probs", nullptr, PluginFieldType::kFLOAT32, 1)); #0
  # "var_seqlen", nullptr, PluginFieldType::kINT32, 1));# 

#  params.d = interface->mHeadSize;

# params.b = B;
# params.h = interface->mNumHeads;
# params.s = S;
# params.d = interface->mHeadSize;  h*d = e

# Input is BxSx3*N*H, output should be BxSxN*H
# 1 in 
# 1 out  
# FusedMHARunnerFP16v2
#
# trtexec --onnx=./mha.onnx --verbose --dumpProfile --separateProfileRun --fp16

import numpy as np
import onnx
from onnx import helper
from onnx import numpy_helper
from onnx import TensorProto

def test():
  inputs = [onnx.helper.make_tensor_value_info("input", TensorProto.FLOAT, shape=(-1, 1, 768, 1, 1))]
  outputs = [onnx.helper.make_tensor_value_info("output", TensorProto.FLOAT, shape=(-1, 1, 256, 1, 1))]

  # kwargs = {'type_id': 1, 'hidden_size': 256, 'num_heads': 4, 'has_mask': 0, 'dq_probs': 0.0}
  kwargs = {'type_id': 1, 'hidden_size': 256, 'num_heads': 8, 'has_mask': 0, 'dq_probs': 0.0, 'var_seqlen': 1}
# CustomQKVToContextPluginDynamic
  nodes = [onnx.helper.make_node("CustomQKVToContextPluginDynamic", name="mha_1", inputs=["input"], outputs=["output"],
    **kwargs)]

  graph = onnx.helper.make_graph(nodes,
                            "mha",
                            inputs,
                            outputs)

  model = helper.make_model(graph)
  onnx.save(model, "/home/igs/rshare/valseqlen_mha.onnx")


test()   
   
