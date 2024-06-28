### 在 trt8510 存在转换问题， 将高维matmul + permute + reshape 进行降维处理    

```
[06/19/2024-19:11:27] [E] Error[2]: [reformatBuilder.cpp::writeGlob::165] Error Code 2: Internal Error (Assertion inputScalesLen >= quantizations.inputs[0].scale.count() failed. )
[06/19/2024-19:11:27] [E] Error[2]: [builder.cpp::buildSerializedNetwork::743] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
[06/19/2024-19:11:27] [E] Engine could not be created from network

```

```
[05/24/2024-15:52:50] [E] Error[2]: [shuffleBuilder.cpp::writeGlob::154] Error Code 2: Internal Error (Assertion outputScalesLen >= quantizations.outputs[0].scale.count() failed. )
[05/24/2024-15:52:50] [E] Error[2]: [builder.cpp::buildSerializedNetwork::743] Error Code 2: Internal Error (Assertion engine != nullptr failed. )
[05/24/2024-15:52:50] [E] Engine could not be created from network

```

+ original version
```
temporal_self_attention.py  
export_temporal_self_attention.py   // 模型导出脚本

trt8510   
trtexec --onnx=tsa_msda_poly.onnx --verbose --best --separateProfileRun --useCudaGraph --dumpProfile 2>&1 |tee v1.log

```

+ optimize version
```
temporal_self_attention_plugin.py   
temporal_self_attention_plugin_bk.py  // 和 temporal_self_attention.py 无差别，仅是支持msda的插件       
export_temporal_self_attention_plugin.py // 模型导出脚本

trt8510   
trtexec --onnx=tsa_msda_plugin_poly.onnx --verbose --best --separateProfileRun --useCudaGraph --dumpProfile 2>&1 |tee v2.log
 
```
