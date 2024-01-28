# -*- coding: utf-8 -*-
# @Time    : 2022/8/29 下午4:55
# @Author  : Teanna
# @File    : trt_inference.py
# @Software: PyCharm

# lix modify based trt_inference.py 20221105

from polygraphy.backend.trt import EngineFromBytes, TrtRunner 
from polygraphy.backend.common import BytesFromPath

from trt_plan import *

class TRT:
    def __init__(self, onnx_file): ## is plan file
        # build_engine(onnx_file)
        engine_file = onnx_file.replace(".onnx", ".plan")
        self.engine = EngineFromBytes(BytesFromPath(engine_file))
        #self.runner = TrtRunner(self.engine)  

    def inference_heads(self, imgs):
      with TrtRunner(self.engine) as runner:
        feed_dict = {"input": imgs.reshape(1, 5, 192, 1024)}
        outputs = runner.infer(feed_dict) 
        #print(outputs.keys()) 
        outputs_vec = []
        outputs_vec.append(outputs["output_seg"])   # output_seg
        outputs_vec.append(outputs["output_prob"])   # output_prob

        return outputs_vec
