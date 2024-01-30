![bn](https://github.com/lix19937/pytorch-cookbook/assets/38753233/cdd3e4dc-bd61-4402-87f7-f2e6001cb3f9)    

**BN 是所有batch的同一通道的内容先求mean 和 std ，因此有几个channel， 就有几个 mean，std **          

* in training    
  ## BN can not fused in conv, has 4 learn paras, but LN has 2 ##  

```
  bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
```
* in infer    
BN 比 LN 在inference的时候快，因为不需要计算 mean 和 variance，直接用 running mean 和 running variance, 同时 BN 可与conv 进行良好融合   


如果将训练的权重参数进行存储，如果model没有进行eval() 或  fuse()   则bn相关参数全部存储到ckpt 中  

```

import torch
import numpy as np

# ref https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html 
torch.manual_seed(0)

# With Learnable Parameters, num_features (int) – C from an expected input of size (N, C, H, W)  
m = torch.nn.BatchNorm2d(num_features=4)
print(m.state_dict())
# OrderedDict([('weight', tensor([1., 1., 1., 1.])), ('bias', tensor([0., 0., 0., 0.])), ('running_mean', tensor([0., 0., 0., 0.])), ('running_var', tensor([1., 1., 1., 1.])), ('num_batches_tracked', tensor(0))])

# Without Learnable Parameters, without `grad_fn=`  
m = torch.nn.BatchNorm2d(num_features=4, affine=False)
print(m.state_dict())
# OrderedDict([('running_mean', tensor([0., 0., 0., 0.])), ('running_var', tensor([1., 1., 1., 1.])), ('num_batches_tracked', tensor(0))])

# [N, C, H, W]
input = torch.randn(2, 4, 3, 3)
output = m(input)

#---------------------------------------------
# y = alpha * (x-mean)/((var+eps)**0.5) + beta  (eps=1e-5, alpha=1, beta=0)
#---------------------------------------------
np_input = input.permute(1, 0, 2, 3) # [C, N, H, W]
mean = torch.mean(np_input, dim=(1, 2, 3), keepdim=True)
var = torch.var(np_input, dim=(1, 2, 3), keepdim=True, unbiased=False)
# print(mean.shape)
output_user= (np_input - mean) / ((var+1e-5)**0.5) * 1 + 0
output_user = output_user.permute(1, 0, 2, 3)
print(torch.max(torch.abs(output_user-output)))

#---------------------------------------------
# the same method of calc mean and var 
#---------------------------------------------
mean = torch.mean(input, dim=(0, 2, 3), keepdim=True)
var = torch.var(input, dim=(0, 2, 3), keepdim=True, unbiased=False)
output_user = (input - mean) / ((var+1e-5)**0.5) * 1 + 0
print(torch.max(torch.abs(output_user-output)))
print(output[:,0,:,:])

# mean, var of all batch's channel_1  
mean_of_channel_1 = torch.mean(input[:, 0, :, :]) 
var_of_channel_1 = torch.var(input[:, 0, :, :], unbiased=False) 
# channel_1 of all out 
out_of_channel1 = (input[:, 0, :, :] - mean_of_channel_1) / ((var_of_channel_1 + 1e-5)**0.5)
print(out_of_channel1)

```
