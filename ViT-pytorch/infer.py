# /**************************************************************
#  * @Copyright: 2021-2022 Copyright 
#  * @Author: lix
#  * @Date: 2022-03-20 11:09:13
#  * @Last Modified by: lix
#  * @Last Modified time: 2022-03-20 11:09:13
#  **************************************************************/

import onnxruntime
import numpy as np
import torch
import math
import onnxsim
from torch.nn import Dropout, Softmax, Linear

from loguru import logger as LOG


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
    with open(file_name + shape_flg, 'a') as f:
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

class ViT_Attention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, batch_first=False, vis=False, is_eval=False):
        super(ViT_Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads  # 8
        self.hidden_size = embed_dim  # 256
        self.attention_head_size = int(embed_dim / self.num_attention_heads)  # 256 / 8 =32
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(embed_dim, self.all_head_size)
        self.key = Linear(embed_dim, self.all_head_size)
        self.value = Linear(embed_dim, self.all_head_size)

        self.out = Linear(embed_dim, embed_dim)
        self.attn_dropout = Dropout(dropout_rate)
        self.proj_dropout = Dropout(dropout_rate)
        self.is_eval = is_eval
        self.softmax = Softmax(dim=-1)
        self.scaling = float(self.attention_head_size) ** -0.5
        self.batch_first = batch_first

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward_vit(self, query, key, value):
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / \
            math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        if not self.is_eval:
            attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        if not self.is_eval:
            attention_output = self.proj_dropout(attention_output)
        # return attention_output, weights
        return attention_output

    def transpose_for_scores_v2(self, x):
        new_x_shape = x.size()[
            :-2] + (self.num_attention_heads, self.attention_head_size)
        LOG.info("{}, {} {}".format(
            x.size()[:-2], self.num_attention_heads, self.attention_head_size))
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)

    ## map to pytorch
    def forward_pt(self, query, key, value):
        assert query.dim() == 3, "only support un-batch_first"
        ## linear 
        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        ## enter into p-ma
        new_x_shape = mixed_query_layer.size()[:-2] + (self.num_attention_heads, self.attention_head_size)
        mixed_query_layer = mixed_query_layer * self.scaling
        mixed_query_layer = mixed_query_layer.view(*new_x_shape)
        query_layer = mixed_query_layer.permute(1, 0, 2)

        # query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores_v2(mixed_key_layer)
        value_layer = self.transpose_for_scores_v2(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))# q v 
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        if not self.is_eval:
            attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (-1, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        ## linear
        attention_output = self.out(context_layer)
        if not self.is_eval:
            attention_output = self.proj_dropout(attention_output)
        # return attention_output, weights
        return attention_output

    ### version: map to nv 
    def forward(self, query, key, value):
        assert query.dim() == 3, "only support un-batch_first"
        L,  N,  E_q = query.size()
        assert N == 1, "batch is 1"

        mixed_query_layer = self.query(query)
        mixed_key_layer = self.key(key)
        mixed_value_layer = self.value(value)

        ## self.attention_head_size = int(E_q / self.num_attention_heads)  # 256 / 8 =32
        ## [L,  N,  E_q] -> [L * N *  self.num_attention_heads, self.attention_head_size]
        aq = mixed_query_layer.view(-1, self.attention_head_size)
        ak = mixed_key_layer.view(-1, self.attention_head_size)
        av = mixed_value_layer.view(-1, self.attention_head_size)

        col_exp = torch.cat((aq, ak, av), 1) 
        PMHA_IN = col_exp.view(L, N, -1, 1, 1)
    
        LOG.info("PMHA_IN:{}".format(PMHA_IN.shape))
        return PMHA_IN
        save_byrow(PMHA_IN.detach().cpu(), "inv.data", fmt = "%.6f")
        
        ## enter into p-ma
        new_x_shape = mixed_query_layer.size()[:-2] + (self.num_attention_heads, self.attention_head_size)
        mixed_query_layer = mixed_query_layer * self.scaling
        mixed_query_layer = mixed_query_layer.view(*new_x_shape)
        query_layer = mixed_query_layer.permute(1, 0, 2)

        # query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores_v2(mixed_key_layer)
        value_layer = self.transpose_for_scores_v2(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))# q v 
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        if not self.is_eval:
            attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (-1, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        LOG.info("context_layer:{}".format(context_layer.shape))
        save_byrow(context_layer.detach().cpu(), "out.data", fmt = "%.6f")
 
        ## linear
        attention_output = self.out(context_layer)
        if not self.is_eval:
            attention_output = self.proj_dropout(attention_output)
        # return attention_output, weights
        return attention_output


class ViT_PMHA(torch.nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_rate, batch_first=False, vis=False, is_eval=False):
        super(ViT_PMHA, self).__init__()
        self.vis = vis
        self.num_attention_heads = num_heads  # 8
        self.hidden_size = embed_dim  # 256
        self.attention_head_size = int(embed_dim / self.num_attention_heads)  # 256 / 8 =32
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.attn_dropout = Dropout(dropout_rate)
        self.proj_dropout = Dropout(dropout_rate)
        self.is_eval = is_eval
        self.softmax = Softmax(dim=-1)
        self.scaling = float(self.attention_head_size) ** -0.5
        self.batch_first = batch_first

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[
            :-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def transpose_for_scores_v2(self, x):
        new_x_shape = x.size()[
            :-2] + (self.num_attention_heads, self.attention_head_size)
        LOG.info("{}, {} {}".format(
            x.size()[:-2], self.num_attention_heads, self.attention_head_size))
        x = x.view(*new_x_shape)
        return x.permute(1, 0, 2)

    def forward(self, query, key, value):
        assert query.dim() == 3, "only support un-batch_first"
        ## linear 
        mixed_query_layer = query
        mixed_key_layer = key
        mixed_value_layer = value

        ## enter into p-ma
        new_x_shape = mixed_query_layer.size()[:-2] + (self.num_attention_heads, self.attention_head_size)
        mixed_query_layer = mixed_query_layer * self.scaling
        mixed_query_layer = mixed_query_layer.view(*new_x_shape)
        query_layer = mixed_query_layer.permute(1, 0, 2)

        key_layer = self.transpose_for_scores_v2(mixed_key_layer)
        value_layer = self.transpose_for_scores_v2(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer, key_layer.transpose(-1, -2))# q v 
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        if not self.is_eval:
            attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(1, 0, 2).contiguous()
        new_context_layer_shape = context_layer.size()[
            :-2] + (-1, self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class PYTORCH_Attention(torch.nn.Module):
    def __init__(self, embed_dim, num_heads,
                 dropout_rate=0.0,
                 bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None,
                 batch_first=False, device=None, dtype=None, is_eval=False):
        super(PYTORCH_Attention, self).__init__()
        self.attn = torch.nn.MultiheadAttention(
            embed_dim, num_heads, dropout_rate)
        self.is_eval = is_eval

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


def savetensor_byrow(x, file_name, fmt="%.6f"):
    shape = x.shape
    leng = len(shape)
    if leng == 1:
        x = x.view(1, -1)
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
        rx = x.view(new_shape)
        with open(file_name + shape_flg, 'a') as f:
            for i in range(new_shape[0]):
                np.savetxt(f, rx[i], fmt=fmt, delimiter=" ")


def helper_onnx(model_file):
    import onnx
    shapes_onnx_filename = model_file + "with_sim.onnx"
    model = onnx.load(model_file)
    model_simp, check = onnxsim.simplify(model)
    assert check, "Simplified ONNX model could not be validated"
    onnx.save(model_simp, shapes_onnx_filename)


def run_model(dummy_input_q, dummy_input_k, dummy_input_v, model, outfile):
    out = model(dummy_input_q, dummy_input_k, dummy_input_v)
    savetensor_byrow(out.detach().cpu(), outfile, "%.6f")
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
    savetensor_byrow(a.detach(), outfile, "%.6f")
    LOG.info("savetensor done")


if __name__ == "__main__":
    rand_seed = 123456
    np.random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)
    torch.cuda.manual_seed_all(rand_seed)

#  query: Query embeddings of shape :math:`(L, E_q)` for unbatched input, :math:`(L, N, E_q)` when ``batch_first=False``
#             or :math:`(N, L, E_q)` when ``batch_first=True``,
# where :math:`L` is the target sequence length,
#       :math:`N` is the batch size,
#       :math:`E_q` is the query embedding dimension ``embed_dim``. d_model

    # (L, N, E_q)
    L = 512
    E_q = 256
    N = 8
    dummy_input_v = torch.randn(L, 1, E_q).cuda()
    dummy_input_q = torch.randn(L, 1, E_q).cuda()
    dummy_input_k = torch.randn(L, 1, E_q).cuda()

    model = ViT_Attention(embed_dim=E_q, num_heads=N,
                          dropout_rate=0.0, vis=False, is_eval=True).cuda()
    model.eval()
    run_model(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/ViT_Attention.outdata")
    run_onnx(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/ViT_Attention.onnx",
             "./output/ViT_Attention_onnx.outdata")

    # model = ViT_PMHA(embed_dim=E_q, num_heads=N, dropout_rate=0.0, is_eval=True).cuda()
    # model.eval()
    # run_model(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/PMHA_onnx.outdata")
    # run_onnx(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/PMHA_onnx.onnx",
    #          "./output/PMHA_onnx.outdata")

    # model = PYTORCH_Attention(
    #     embed_dim=E_q, num_heads=N, dropout_rate=0.0, is_eval=True).cuda()
    # model.eval()
    # run_model(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/PT_Attention.outdata")
    # run_onnx(dummy_input_q, dummy_input_k, dummy_input_v, model, "./output/PT_Attention.onnx",
    #          "./output/PT_Attention_onnx.outdata")
