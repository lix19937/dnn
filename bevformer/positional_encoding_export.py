
# models/utils/positional_encoding.py

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
