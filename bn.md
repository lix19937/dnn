![bn](https://github.com/lix19937/pytorch-cookbook/assets/38753233/cdd3e4dc-bd61-4402-87f7-f2e6001cb3f9)    



* in training    
bn not fused in conv,
```
bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
```

如果将训练的权重参数进行存储，如果model没有进行eval() 或  fuse()   则bn相关参数全部存储到ckpt 中  

```
import torch
import numpy as np

torch.manual_seed(0)
# With Learnable Parameters
m = torch.nn.BatchNorm2d(4)
# Without Learnable Parameters, without grad_fn=
m = torch.nn.BatchNorm2d(4, affine=False)

# [N, C, H, W]
input = torch.randn(2, 4, 3, 3)
output = m(input)

#---------------------------------------------
# y = alpha * (x-mean)/((var+eps)**0.5) + beta  (eps=1e-5, alpha=1, beta=0)
#---------------------------------------------
np_input = input.permute(1,0,2,3) # [C, N, H, W]
mean = torch.mean(np_input, dim=(1,2,3), keepdim=True)
var = torch.var(np_input, dim=(1,2,3), keepdim=True, unbiased=False)
# print(mean.shape)
output_user= (np_input - mean) / ((var+1e-5)**0.5) * 1 + 0
output_user = output_user.permute(1,0,2,3)
print(torch.max(torch.abs(output_user-output)))

#---------------------------------------------
# the same method of calc mean and var 
#---------------------------------------------
mean = torch.mean(input, dim=(0,2,3), keepdim=True)
var = torch.var(input, dim=(0,2,3), keepdim=True, unbiased=False)
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
