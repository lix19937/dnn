## ln 的cuda实现
*  two pass

*  one pass

*  welford    



```
import torch
import numpy as np

# ref https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html  
torch.manual_seed(0)

# LN特别适合处理变长数据, 因为是对 hidden 维度做操作,和句子长度和batch大小无关   
batch, sentence_length, embedding_dim = 2, 4, 3
input = torch.randn(batch, sentence_length, embedding_dim)

# With Learnable Parameters
m = torch.nn.LayerNorm(embedding_dim)
print(m.state_dict())
# OrderedDict([('weight', tensor([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])), ('bias', tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]))])

# Without Learnable Parameters, without `grad_fn=`
m = torch.nn.LayerNorm(embedding_dim, elementwise_affine=False)
print(m.state_dict())
# OrderedDict()  

output = m(input)

#---------------------------------------------
# y = alpha * (x-mean)/((var+eps)**0.5) + beta  (eps=1e-5, alpha=1, beta=0)
#---------------------------------------------
mean = torch.mean(input, dim=(2), keepdim=True)
var = torch.var(input, dim=(2), keepdim=True, unbiased=False)
print(mean.shape)

# 注意torch.var https://discuss.pytorch.org/t/torch-var-and-torch-std-return-nan/38884/2   
output_user = (input - mean) / ((var+1e-5)**0.5) * 1 + 0
print(torch.max(torch.abs(output_user - output)))

input = torch.randn(2, 4, 3, 3)
m = torch.nn.LayerNorm([4, 3, 3])
output = m(input)
print(output.shape)


```

ref  https://changyaochen.github.io/welford/   

