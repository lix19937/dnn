
# see https://github.com/lix19937/dnn-cookbook/blob/main/gpt2/gpt2.md#gpt2attention   

import torch
import numpy as np

rand_seed = 123456
np.random.seed(rand_seed)
torch.manual_seed(rand_seed)
torch.cuda.manual_seed_all(rand_seed)

def savetensor_byrow(x, file_name, fmt = "%.6f", delimiter=" "):
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
    np.savetxt(file_name + shape_flg, x, fmt=fmt, delimiter=delimiter)   
  if leng > 2:
    cs = 1
    for i in range(leng - 2):
      cs = cs*shape[i]

    new_shape = (cs, shape[-2], shape[-1])
    rx = x.reshape(new_shape)
    with open(file_name + shape_flg, 'w') as f:
      for i in range(new_shape[0]):
        np.savetxt(f, rx[i], fmt=fmt, delimiter=delimiter)

def test():
  query = torch.arange(0, 1*2*4*4).view(1, 2, 4, 4).float()
  key = torch.randn(1, 2, 4, 4).float()
  attn_weights = torch.matmul(query, key.transpose(-1, -2)) # [1, 2, 4, 4]

  savetensor_byrow(attn_weights, "attn_weights-v1-" + ".log")

  #------------------------------------------------------
  past_key = key
  #------------------------------------------------------

  curr_query = torch.randn(1, 2, 1, 4).float()   
  query = torch.cat([query, curr_query], dim=-2) #step=1  [1, 2, 5, 4]

  curr_key = torch.randn(1, 2, 1, 4).float()   
  key = torch.cat([key, curr_key], dim=-2) # [1, 2, 5, 4]

  attn_weights = torch.matmul(query, key.transpose(-1, -2)) # [1, 2, 5, 5]

  savetensor_byrow(attn_weights, "attn_weights-v1-" + ".log")

  #------------------------------------------------------
  key = torch.cat([past_key, curr_key], dim=-2) # [1, 2, 5, 4]
  attn_weights = torch.matmul(curr_query, key.transpose(-1, -2)) # [1, 2, 4, 4]

  savetensor_byrow(attn_weights, "attn_weights-v2-" + ".log")
  #------------------------------------------------------

test()
