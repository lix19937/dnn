* in training
bn not fused in conv,
```
bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias
```

如果将训练的权重参数进行存储，如果model没有进行eval() 或  fused_bn()   则bn相关参数全部存储到ckpt 中  
