
## 如果 torch layer 无法直接转成 onnx op，可以手工使用基本torch layer 实现该layer    
```
背景：本来pytorch是实现了逆矩阵的函数的，1.8之前是torch.inverse()，1.8是在torch.linalg.inv()

但是，cuda不支持逆矩阵inverse，因此onnx当然也没实现inverse 
但是我们训练的模型需要移植到onnx甚至cuda就会变得很困难，甚至你想在pytorch里面使用model.half()进行半精度运算的时候，就会报错说，cuda不支持inverse 
因此，只能换个思路，自己实现这个算子，使用更为常见的算子来进行替代inverse 
pytorch和numpy的源码很难找到inverse的具体实现，因此我只能另想办法，直到看到这篇文章，人家使用numpy进行实现的，那我可以直接改为pytorch版的   
```
```
import torch
from torch.linalg import det

def cof1(M,index):
    zs = M[:index[0]-1,:index[1]-1]
    ys = M[:index[0]-1,index[1]:]
    zx = M[index[0]:,:index[1]-1]
    yx = M[index[0]:,index[1]:]
    s = torch.cat((zs,ys),axis=1)
    x = torch.cat((zx,yx),axis=1)
    return det(torch.cat((s,x),axis=0))

def alcof(M,index):
    return pow(-1,index[0]+index[1])*cof1(M,index)

def adj(M):
    result = torch.zeros((M.shape[0],M.shape[1]))
    for i in range(1,M.shape[0]+1):
        for j in range(1,M.shape[1]+1):
            result[j-1][i-1] = alcof(M,[i,j])
    return result

def invmat(M):
    return 1.0/det(M)*adj(M)

M = torch.FloatTensor([[1,2,-1],[2,3,4],[3,1,2]])
print(invmat(M))
print(torch.inverse(M))
```
