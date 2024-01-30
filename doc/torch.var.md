无偏估计  

https://discuss.pytorch.org/t/torch-var-and-torch-std-return-nan/38884/5   
torch.var两种方差都可以计算，这取决于一个参数，即unbiased，无偏的意思。默认值为true，也就是说，默认的目的是样本估计总体，计算的是**样本方差**。

如果是非无偏估计，则是计算**总体方差**       

```py

import torch
a=torch.tensor([1.0, -1])
ret = torch.var(a)  #  分母除以的是1.

print(ret)
# tensor(2.)

a=torch.tensor([1.0, -1])
ret = torch.var(a, unbiased=False) #  分母除以的是2.
print(ret)
#tensor(1.)

```
