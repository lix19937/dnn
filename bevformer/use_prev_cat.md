
避免 onnx 中有条件分支（需要script 方式导出）  

```
x1 = [  1, 512, 200, 200]
w1 = [256, 512, 3,   3]
y1 = x1 * w1  

x2 = x1[  1, 256:, 200, 200]
w2 = x2[256, 256:, 3,   3]
y2 = x2 * w2  
```

```py
def test():
    torch.random.manual_seed(0)

    kernel_data = torch.randn(256, 768, 3, 3)
    bias_data = torch.randn(256)


    x1 = torch.randn(1, 768, 200, 200)
    x1[:, 0:512, :, :] = 0
    x2 = x1[:, 512:, :, :]

    conv = nn.Conv2d(in_channels=768, out_channels=256, kernel_size=(3, 3),stride=1, padding=1, padding_mode='zeros', 
        bias=True)
    conv.weight = nn.Parameter(kernel_data)
    conv.bias = nn.Parameter(bias_data)

    ret = conv(x1)

    conv_s1 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3, 3),stride=1, padding=1, padding_mode='zeros', 
        bias=True)
    conv_s1.weight = nn.Parameter(kernel_data[:, 512:, :, :])
    conv_s1.bias = nn.Parameter(bias_data)

    ret_s = conv_s1(x2)  

    logger.info(f"check {torch.equal(ret, ret_s)}")

    logger.info("done")
```
