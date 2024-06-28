# Copyright (c) OpenMMLab. All rights reserved.

import torch
import torch.nn as nn
import numpy as np

from temporal_self_attention import TemporalSelfAttention

from loguru import logger 

############################################################################

query= torch.randn(1, 400, 256, dtype=torch.float32) 
key=None
value=None
identity=None
query_pos= torch.randn(1, 400, 256, dtype=torch.float32) 
key_padding_mask=None
reference_points=torch.randn(2, 400, 1, 2, dtype=torch.float32)
# spatial_shapes =torch.tensor([[20, 20]], dtype=torch.int64)     #  torch.Size([1, 2])  torch.int64  
# level_start_index=torch.tensor([0], dtype=torch.int64)

nn_model = TemporalSelfAttention(
    embed_dims=256,
    num_heads=8,
    num_levels=1,
    num_points=4,
    num_bev_queue=2,
    im2col_step=64,
    dropout=0.1,
    batch_first=True,
    norm_cfg=None,
    init_cfg=None
)  

ref = nn_model(                
    query,
    key,
    value,
    identity,
    query_pos,
    key_padding_mask,
    reference_points,
    # spatial_shapes,
    # level_start_index
    )

logger.info('------------------------------')

output_file = 'tsa_msda.onnx' 
torch.onnx.export(
    nn_model,
    (query,
    key,
    value,
    identity,
    query_pos,
    key_padding_mask,
    reference_points,
    # spatial_shapes,
    # level_start_index
    ),
    output_file,
    # input_names=['query', 'query_pos', 'reference_points'],
    export_params=True,
    keep_initializers_as_inputs=True,
    do_constant_folding=True,
    # enable_onnx_checker=True, 
    verbose=False,
    opset_version=13
)
logger.info("export done")

########################
output_file_simp = output_file.replace(".onnx", "_simp.onnx")    

from onnxsim import simplify, model_info 
import onnx
onnx_model = onnx.load(output_file)   
model_simp, check_ok = simplify(onnx_model)
assert check_ok, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_file_simp)

if check_ok:
    logger.info("Finish! Here is the difference:")
    model_info.print_simplifying_info(onnx_model, model_simp)
else:
    logger.error(
        'Check failed. Please be careful to use the simplified model, or try specifying "--skip-fuse-bn" or "--skip-optimization" (run "onnxsim -h" for details).'
    )
    logger.error("Here is the difference after simplification:")
    model_info.print_simplifying_info(onnx_model, model_simp)
    exit(1)

logger.info('simplify done')

########################
output_file_poly = output_file.replace(".onnx", "_poly.onnx")    

cmd = "polygraphy surgeon sanitize --fold-constants " + str(output_file)  + " -o " + output_file_poly
import subprocess
ret, val = subprocess.getstatusoutput(cmd)
logger.info(f'polygraphy returncode:{ret}\n{val}')
logger.info('polygraphy done')

########################
# validate the converted onnx operation
# import onnxruntime as ort

# sess = ort.InferenceSession(output_file_simp)
# outputs = sess.run(None, {'query': query.numpy(), 
#                           'query_pos': query_pos.numpy(), 
#                           'reference_points': reference_points.numpy()})
# out_onnx = outputs[0]


# print(np.allclose(ref.detach().numpy(), out_onnx, rtol=1e-3, atol=1e-1))
# t = ref.detach().numpy() - out_onnx
# print(t.min(), t.max(), abs(t).mean(), abs(t).var())
