
# models/utils/positional_encoding.py

# https://zhuanlan.zhihu.com/p/633836459   

import torch

from mmdet.models.utils.positional_encoding import LearnedPositionalEncoding

class positional_encoding_nodel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.positional_encoding = LearnedPositionalEncoding(
            num_feats=128,
            row_num_embed=200,
            col_num_embed=200)

    def forward(self, bev_mask):
        # bev_mask = torch.zeros(bs, self.bev_h, self.bev_w)
        bev_pos = self.positional_encoding(bev_mask)     
        return bev_pos
    
bev_mask = (torch.randn(1, 200, 200) > 0.1)  # torch.bool
nn_model = positional_encoding_nodel()

output_file = 'positional_encoding.onnx' 
torch.onnx.export(
    nn_model,
    (bev_mask),
    output_file,
    input_names=['input'],
    export_params=True,
    keep_initializers_as_inputs=True,
    do_constant_folding=True,
    verbose=True,
    opset_version=13,
)
print("export done")
t1 = nn_model(bev_mask).detach().numpy()
print(t1.shape)

########################  
output_file_simp = 'positional_encoding_simp.onnx' 

import onnxruntime as ort
from onnxsim import simplify
import onnx
onnx_model = onnx.load(output_file)   
model_simp, check = simplify(onnx_model)
assert check, "Simplified ONNX model could not be validated"
onnx.save(model_simp, output_file_simp)
print('simplify onnx done')

sess = ort.InferenceSession(output_file_simp)
outputs = sess.run(None, {'input': bev_mask.detach().numpy()})
t2 = outputs[0]

########################
output_file_poly = 'positional_encoding_poly.onnx' 

from polygraphy.backend.onnx import fold_constants

model_poly_folded = fold_constants(onnx_model)
onnx.save(model_poly_folded, output_file_poly)

sess = ort.InferenceSession(output_file_poly)
outputs = sess.run(None, {'input': bev_mask.detach().numpy()})
t3 = outputs[0]

print((t1==t2).all())
print("\n-----------------------\n")
print((t3==t2).all())

########################
output_file_poly_sp = 'positional_encoding_poly_sp.onnx' 

cmd = "polygraphy surgeon sanitize --fold-constants " + str(output_file)  + " -o " + output_file_poly_sp
import subprocess
ret, val = subprocess.getstatusoutput(cmd)
print('returncode:', ret)

# polygraphy surgeon sanitize --fold-constants positional_encoding.onnx  -o folded_model2.onnx

# trtexec  --verbose --onnx=./folded_model.onnx --dumpProfile
# trtexec  --verbose --onnx=./positional_encoding.onnx --dumpProfile

# https://github.com/zw0610/zw0610.github.io
# https://www.stubbornhuang.com/2962/

# https://onnxruntime.ai/docs/reference/compatibility.html#onnx-opset-support   
# https://mmdetection.readthedocs.io/en/v2.14.0/_modules/mmdet/models/utils/positional_encoding.html

'''
Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
Warning: Constant folding - Only steps=1 can be constant folded for opset >= 10 onnx::Slice op. Constant folding not applied.
Traceback (most recent call last):
  File "./tools/test.py", line 290, in <module>
    main()
  File "./tools/test.py", line 267, in main
    torch.onnx.export(
  File "/home/lix/anaconda3/envs/bevformer-v2/lib/python3.8/site-packages/torch/onnx/__init__.py", line 275, in export
    return utils.export(model, args, f, export_params, verbose, training,
  File "/home/lix/anaconda3/envs/bevformer-v2/lib/python3.8/site-packages/torch/onnx/utils.py", line 88, in export
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
  File "/home/lix/anaconda3/envs/bevformer-v2/lib/python3.8/site-packages/torch/onnx/utils.py", line 689, in _export
    _model_to_graph(model, args, verbose, input_names,
  File "/home/lix/anaconda3/envs/bevformer-v2/lib/python3.8/site-packages/torch/onnx/utils.py", line 463, in _model_to_graph
    graph = _optimize_graph(graph, operator_export_type,
  File "/home/lix/anaconda3/envs/bevformer-v2/lib/python3.8/site-packages/torch/onnx/utils.py", line 223, in _optimize_graph
    torch._C._jit_pass_onnx_graph_shape_type_inference(graph, params_dict, _export_onnx_opset_version)
RuntimeError: unexpected tensor scalar type
'''
