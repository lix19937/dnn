池化
```
压缩图片，保留重要信息
```

RELU
```
将feature out负数转为0
```

图像进入dnn，特征会逐层变得更复杂，featuremap尺寸一般变得更小。最终，前面的layers包含了简单的特征，如棱角或点；越后面的层则包含一些复杂的特征，如形状或图案。这些高端的特征通常变得很好辨识。  
初始，一个未经训练的dnn，其中权重和偏置都是随机决定的。
对于分类任务：  
训练样本和标签分批进入初始网络，每一批样本经前向推导，最终到达分类层得到类别分配，与对应标签值比较，计算loss（这一批样本分类的误判，即辨识误差），指导我们怎样才算好的权重和偏置参数。通过调整参数，让loss降低。在每次调整后，这些参数被微调，loss也会重新计算，成功降低loss的调整将被保留。所以当调整卷积参数以后，可以得到一组更擅长于判断当前样本的权重参数。重复上述过程，辨识更多已标记的样本。训练过程中，存在个别样本被误判，但这些样本中共通的特征会被学习到（反映在卷积核参数中）。如果有足够多的已标记样本，这些权重参数最后会趋近一个辨识大多数样本的稳定状态。   

超参数  
```
卷积核的size 与个数以及stride，池化层的size与stride，偏置
```

coco数据集说明
https://zhuanlan.zhihu.com/p/309549190   
https://github.com/rwightman/pytorch-image-models


[finetune_alexnet_with_tensorflow-cv.zip](https://github.com/lixwy/dl_tutorial/files/6682578/finetune_alexnet_with_tensorflow-cv.zip)


[中文版_神经网络与深度学习Michael.pdf](https://github.com/lixwy/dl_tutorial/files/6681566/_.Michael.pdf)  
http://neuralnetworksanddeeplearning.com/chap1.html  
https://towardsdatascience.com/an-introduction-to-convolutional-neural-networks-eb0b60b58fd7  
https://towardsdatascience.com/a-beginners-guide-to-convolutional-neural-networks-cnns-14649dbddce8"  
https://machinethink.net/blog/convolutional-neural-networks-on-the-iphone-with-vggnet"  VggNet for embedded systems  
https://machinethink.net/blog/googles-mobile-net-architecture-on-iphone  Google's mobilenet on embedded systems 
https://analyticsvidhya.com/blog/2019/02/tutorial-semantic-segmentation-google-deeplab  Google deeplab for semantic segmentation  
https://www.analyticsvidhya.com/blog/2019/04/introduction-image-segmentation-techniques-python  Image segmentation techniques  
https://zhuanlan.zhihu.com/p/350898311
