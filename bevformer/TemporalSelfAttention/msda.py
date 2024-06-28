
import torch 

class Identity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value

    @staticmethod
    def symbolic(
        g,
        value):
        return g.op(  
            "Identity",
            value)

_identity_ = Identity.apply

class MultiScaleDeformableAttnFunction2(torch.autograd.Function):
    @staticmethod
    def forward(ctx,  
                value, 
                spatial_shapes,
                level_start_index, 
                sampling_locations,
                attention_weights, 
                im2col_step):
        out = torch.randn(value.shape[0], attention_weights.shape[1], value.shape[2], value.shape[3])
        return out

    @staticmethod
    def symbolic(
        g,
        value, 
        spatial_shapes,
        level_start_index, 
        sampling_locations,
        attention_weights, 
        im2col_step
        ):
        return g.op(  
            "TRT::MultiscaleDeformableAttnPlugin_TRT",
            # "MultiScaleDeformableAttn_TRT2_HM",
            value, 
            spatial_shapes,
            level_start_index, 
            sampling_locations,
            attention_weights, 
            )

multi_scale_deformable_attn = MultiScaleDeformableAttnFunction2.apply

# class MSDA(torch.nn.Module):
#     def __init__(self):
#         super().__init__()

#     def forward(self, 
#                 value, 
#                 spatial_shapes,
#                 level_start_index, 
#                 sampling_locations,
#                 attention_weights, 
#                 im2col_step):
#         return multi_scale_deformable_attn(
#             value, 
#             spatial_shapes,
#             level_start_index, 
#             sampling_locations,
#             attention_weights, 
#             im2col_step)

## 1336 from fpn (related to img shape )
## 2304 / 9216 from bev shape 
# bev_query = 9216

# value = torch.randn(6, 1336, 8, 32)  
# spatial_shapes = torch.tensor([[20, 50], [10, 25], [ 5, 13], [3,  7]], dtype=torch.int64)  
# level_start_index = torch.tensor([0, 1000, 1250, 1315], dtype=torch.int64)  
# sampling_locations =torch.randn(6, 2304, 8, 4, 8, 2)  
# attention_weights =torch.randn(6, 2304, 8, 4, 8)  
# im2col_step = 64 #torch.tensor([64], dtype=torch.int64)  

# model = MSDA().eval()

# torch.onnx.export(model, (
#                             value, 
#                             spatial_shapes,
#                             level_start_index, 
#                             sampling_locations,
#                             attention_weights, 
#                             im2col_step), 
#                           'msda_hm.onnx',
#                           export_params=True,
#                           keep_initializers_as_inputs=True,
#                           do_constant_folding=True,
#                           enable_onnx_checker=True, 
#                           verbose=True,
#                           opset_version=13) 

