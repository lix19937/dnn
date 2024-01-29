# /**************************************************************
#  * @Copyright: 2021-2022 Copyright SAIC
#  * @Author: lijinwen
#  * @Date: 2022-03-20 11:09:13
#  * @Last Modified by: lijinwen
#  * @Last Modified time: 2022-03-20 11:09:13
#  **************************************************************/

import onnxruntime
import numpy as np
import torch
import math
import onnxsim
from torch.nn import Dropout, Softmax, Linear
import torch.nn.functional as F

from loguru import logger as LOG

# v2 un-batch first 

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
    with open(file_name + shape_flg, 'w') as f:
      for i in range(new_shape[0]):
        np.savetxt(f, rx[i], fmt=fmt, delimiter=" ")

def test():
  # [3, 8] --> [3, 2, 4]
  x = torch.arange(0, 24).view(6,4)
  y = torch.arange(0, 24).view(6,4) + 100
  z = torch.arange(0, 24).view(6,4) + 200

  print(x)
  print(y)
  w= torch.cat((x,y,z), 1)
 
  print('\n===============\n')
  print(w.shape,'\n', w)

  r = w.view(3, -1)
  print(r.shape,'\n', r)
  exit()

### https://zhuanlan.zhihu.com/p/414511434
### https://developer.nvidia.com/blog/nlu-with-tensorrt-bert/
### https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py#L304
class BertSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, batch_first=False, vis=False, is_eval=False):
        super(BertSelfAttention, self).__init__()
        hidden_size, num_attention_heads, attention_probs_dropout_prob = embed_dim, num_heads, dropout_rate
        if hidden_size % num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(hidden_size, self.all_head_size)
        self.key = torch.nn.Linear(hidden_size, self.all_head_size)
        self.value = torch.nn.Linear(hidden_size, self.all_head_size)
        self.dropout = torch.nn.Dropout(attention_probs_dropout_prob)
        self.is_eval = is_eval        

    def transpose_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(1)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).transpose(0, 1)
        return x

    def transpose_key_for_scores(self, x):
        # seq: x.size(0), bsz: x.size(1)
        x = x.view(x.size(0), x.size(1) * self.num_attention_heads, self.attention_head_size).permute(1, 2, 0)
        return x

    def forward(self, hidden_states, attention_mask):
        # (seq, bsz, hidden)
        batch_size = hidden_states.size(1)
        seq_length = hidden_states.size(0)

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        mixed_query_layer = (hidden_states)
        mixed_key_layer = (hidden_states)
        mixed_value_layer = (hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_key_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.bmm(query_layer, key_layer)

        # (bsz, heads, seq, seq)
        attention_scores = attention_scores.view(batch_size, self.num_attention_heads, seq_length, seq_length)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/modeling.py#L853
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = F.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (bsz, heads, seq, seq)
        if not self.is_eval:
          attention_probs = self.dropout(attention_probs)
        attention_probs = attention_probs.view(batch_size * self.num_attention_heads, seq_length, seq_length)

        context_layer = torch.bmm(attention_probs, value_layer)
        context_layer = context_layer.transpose(0, 1).contiguous()
        # (seq, bsz, hidden)
        context_layer = context_layer.view(seq_length, batch_size, self.all_head_size)
        return context_layer


class ViT_Attention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, weight_list = [],  batch_first=False, vis=False, is_eval=False):
        super(ViT_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads  # 8
        self.hidden_size = embed_dim  # 256
        self.attention_head_size = int(embed_dim / self.num_attention_heads)  # 256 / 8 =32
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(embed_dim, self.all_head_size) ###
        self.key = Linear(embed_dim, self.all_head_size)   ###
        self.value = Linear(embed_dim, self.all_head_size) ###
        self.out = Linear(embed_dim, embed_dim) #####

# model.query_weight.t(), model.key_weight.t(),             model.value_weight.t(), 
# model.query_bias,       model.key_bias, model.value_bias, model.out_proj_weight.t()
#
        if len(weight_list) > 0:
          print("-------------------------------------->>")
          self.query.weight = torch.nn.Parameter(weight_list[0])
          self.key.weight   = torch.nn.Parameter(weight_list[1])
          self.value.weight = torch.nn.Parameter(weight_list[2])####

          self.query.bias = torch.nn.Parameter(weight_list[3])
          self.key.bias   = torch.nn.Parameter(weight_list[4])
          self.value.bias = torch.nn.Parameter(weight_list[5])#########

          self.out.weight = torch.nn.Parameter(weight_list[6])
          self.out.bias   = torch.nn.Parameter(weight_list[7])

        self.attn_dropout = Dropout(dropout_rate)
        self.proj_dropout = Dropout(dropout_rate)
        self.is_eval = is_eval
        self.softmax = Softmax(dim=-1)
        self.scaling = 1/ math.sqrt(self.attention_head_size) # float(self.attention_head_size) ** -0.5
        self.scaling_r = math.sqrt(self.attention_head_size) 
        self.batch_first = batch_first

    def transpose_for_scores(self, x): # x shape[512, 1, 256]
        new_x_shape = x.size(0) * + (x.size(1) * self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape) # [512, 8, 32]
        return x.permute(1, 0, 2) # [8, 512, 32]

    ## map to vit github repo
    def forward_vit(self, query, key, value):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores * self.scaling #1/ math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        if not self.is_eval:
            attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (1, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        if not self.is_eval:
            attention_output = self.proj_dropout(attention_output)
        # return attention_output, weights
        return attention_output

    ## optimize of `forward_vit`, just scale in ahead of permute , Fully align to pytorch impl 
    def forward_scale_in_ahead(self, query, key, value):
        assert query.dim() == 3, "only support un-batch_first"
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer/ self.scaling_r #* self.scaling

        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))# q k^T 
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        if not self.is_eval:
            attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer) # v
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (1, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        attention_output = self.out(context_layer)
        if not self.is_eval:
            attention_output = self.proj_dropout(attention_output)
        # return attention_output, weights
        return attention_output

    def forward(self, query, key, value):
      return self.forward_vit(query, key, value)


class PYTORCH_Attention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads,
                 dropout_rate=0.0,
                 bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None):
        super(PYTORCH_Attention, self).__init__()
        self.attn = torch.nn.MultiheadAttention(embed_dim, num_heads, dropout_rate, 
          bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None, batch_first=False)

        # print(self.attn.state_dict().keys())
        # odict_keys(['in_proj_weight', 'in_proj_bias', 'out_proj.weight', 'out_proj.bias'])
        # for k,v in self.attn.state_dict().items():
        #   print(k, v.shape)
        # in_proj_weight torch.Size([768, 256]) # [3 * embed_dim, embed_dim]
        # in_proj_bias torch.Size([768])
        # out_proj.weight torch.Size([256, 256])
        # out_proj.bias torch.Size([256])
        # be careful linear is diff with torch.matmul !!!    in_proj_weight shape need transpose 
        ##########################################################################################
        in_proj_weight_printf_T = self.attn.in_proj_weight.permute(1, 0)
        self.query_weight = in_proj_weight_printf_T[:, 0:embed_dim]
        self.key_weight = in_proj_weight_printf_T[:, embed_dim:embed_dim*2]
        self.value_weight = in_proj_weight_printf_T[:, embed_dim*2:embed_dim*3]
        
        in_proj_bias_printf_T = self.attn.in_proj_bias 
        self.query_bias = in_proj_bias_printf_T[0:embed_dim]
        self.key_bias = in_proj_bias_printf_T[embed_dim:embed_dim*2]
        self.value_bias = in_proj_bias_printf_T[embed_dim*2:]

        out_proj_weight_printf_T = self.attn.out_proj.weight.permute(1, 0)
        self.out_proj_weight = out_proj_weight_printf_T

        out_proj_bias_printf_T = self.attn.out_proj.bias 
        self.out_proj_bias = out_proj_bias_printf_T

      #################################################
        w_q, w_k, w_v = self.attn.in_proj_weight.chunk(3)
        b_q, b_k, b_v = self.attn.in_proj_bias.chunk(3)
        o_w = self.attn.out_proj.weight
        o_b = self.attn.out_proj.bias 
        self.weight_list = [w_q, w_k, w_v, b_q, b_k, b_v, o_w, o_b]
      ##########################################################################################

    def forward(self, query, key, value):
      return self.attn(
        query=query,
        key=key,
        value=value,
        key_padding_mask=None, 
        need_weights=True, 
        attn_mask=None)[0]


class ONNXModel():
    def __init__(self, onnx_path):
        self.onnx_session = onnxruntime.InferenceSession(
            onnx_path, providers=['CPUExecutionProvider', 'CUDAExecutionProvider'])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        LOG.info("input_name:{}".format(self.input_name))
        LOG.info("output_name:{}".format(self.output_name))

    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name

    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_feed(self, input_name, input_numpy):
        LOG.info("input_name:{}".format(input_name))
        LOG.info("input_numpy length:{}".format(len(input_numpy)))

        assert len(input_name) == len(input_numpy)
        input_feed = {}
        for i in range(len(input_name)):
            input_feed[input_name[i]] = input_numpy[i]
        return input_feed

    def forward(self, input_numpy):
        input_feed = self.get_input_feed(self.input_name, input_numpy)
        results = self.onnx_session.run(
            self.output_name, input_feed=input_feed)
        return results, self.output_name


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def helper_onnx(model_file):
    import onnx
    shapes_onnx_filename = model_file + "with_sim.onnx"
    model = onnx.load(model_file)
    model_simp, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, shapes_onnx_filename)


def run_model(dummy_input_q, dummy_input_k, dummy_input_v, model, outfile):
    # import pdb;pdb.set_trace()
    out = model(dummy_input_q, dummy_input_k, dummy_input_v)
    save_byrow(out.detach().cpu(), outfile, "%.6f")
    LOG.info("savetensor done")


def run_model_bert(dummy_inputs, model, outfile):
    out = model(*dummy_inputs)
    save_byrow(out.detach().cpu(), outfile, "%.6f")
    LOG.info("savetensor done")


def run_onnx(dummy_input_q, dummy_input_k, dummy_input_v, model, out_onnx, outfile):
    onnx_path = out_onnx
    torch.onnx.export(model,
                      (dummy_input_q, dummy_input_k, dummy_input_v),
                      onnx_path,
                      verbose=False,
                      opset_version=11,
                      enable_onnx_checker=True,
                      do_constant_folding=True)
    LOG.info("export done")

    helper_onnx(onnx_path)
    LOG.info("helper_onnx done")

    net = ONNXModel(onnx_path)
    q = to_numpy(dummy_input_q)
    k = to_numpy(dummy_input_k)
    v = to_numpy(dummy_input_v)

    out, _ = net.forward([q, k, v])
    LOG.info("forward done")

    a = torch.from_numpy(out[0])
    save_byrow(a.detach(), outfile, "%.6f")
    LOG.info("savetensor done")


if __name__ == "__main__":
    rand_seed = 123456
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

#  query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
#             or :math:`(N, L, E_q)` when ``batch_first=True``,
# where :math:`L` is the target sequence length,
#       :math:`N` is the batch size,
#       :math:`E_q` is the query embedding dimension ``embed_dim``.

    # (L, N, E_q)
    L = 512
    E_q = 256
    N = 8
    dummy_input_q = torch.randn(L, 1, E_q).cuda()
    dummy_input_k = torch.randn(L, 1, E_q).cuda()
    dummy_input_v = torch.randn(L, 1, E_q).cuda()
   
    pymodel = PYTORCH_Attention(embed_dim=E_q, num_heads=N, dropout_rate=0.0).cuda()
    
    pymodel.eval()
    run_model(dummy_input_q, dummy_input_k, dummy_input_v, pymodel, "./output/TORCH_Attention.outdata")
    torch.onnx.export(pymodel,
                      (dummy_input_q, dummy_input_k, dummy_input_v),
                      "output/pt_self_atten.onnx",
                      verbose=False,
                      opset_version=11,
                      enable_onnx_checker=True,
                      do_constant_folding=True)
    LOG.info("export done")
    helper_onnx("output/pt_self_atten.onnx")
    LOG.info("helper_onnx done")

    ####################################################################################################
    model = ViT_Attention(embed_dim=E_q, num_heads=N, dropout_rate=0.0, 
                    weight_list=pymodel.weight_list,
                    vis=False, is_eval=True).cuda()
    model.eval()
    run_model(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/vit_self_atten_v2.outdata")
    torch.onnx.export(model,
                      (dummy_input_q, dummy_input_k, dummy_input_v),
                      "output/vit_self_atten_v2.onnx",
                      input_names=("query", "key", "value"),
                      verbose=False,
                      opset_version=11,
                      enable_onnx_checker=True,
                      do_constant_folding=True)
    LOG.info("export done")
    helper_onnx("output/vit_self_atten_v2.onnx")
    LOG.info("helper_onnx done")
    exit(0)
    model = BertSelfAttention(embed_dim=E_q, num_heads=N, dropout_rate=0.0, vis=False, is_eval=True).cuda()
    model.eval()

    attention_mask = torch.ones_like(dummy_input_q)
    token_type_ids = torch.zeros_like(dummy_input_q)
    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
    extended_token_type_ids = token_type_ids.unsqueeze(1).unsqueeze(2)

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    print(extended_attention_mask.shape)# （batch_size, 1, 1, seq_length）
    print(torch.equal(extended_token_type_ids, extended_attention_mask))

    run_model_bert([dummy_input_q, extended_attention_mask], model, "./output/BERT_Attention-bert.outdata")

    torch.onnx.export(model,
                      (dummy_input_q, extended_attention_mask),
                      "output/bert_self_atten.onnx",
                      verbose=False,
                      opset_version=11,
                      enable_onnx_checker=True,
                      do_constant_folding=True)
    LOG.info("export done")
    helper_onnx("output/bert_self_atten.onnx")
    LOG.info("helper_onnx done")
