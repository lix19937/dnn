backbone + head    

https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-linux-installation/index.html
https://developer.nvidia.com/docs/drive/drive-os/6.0.6/public/drive-os-tensorrt/developer-guide/index.html

账号：gw00348951    
642061776leeLX-

https://github.com/NVIDIA/TensorRT/issues/2735

`DETR  ->  DETR3D  ->  BEVFormer   ->  BEVFormer v2`         

bev视角下可以实现端到端的目标检测、语义分割、轨迹预测等各项任务。（目标检测任务，车道线语义分割类任务，场景分类任务）

bev视角的好处    
+ 基于视觉的检测属于前向视角，在成像的过程中天然的存在近大远小的特点。例如车道线，在成像的图片上是一条相交的直线，而在俯视角下直接预测为直线更接近现实场景；
同时在BEV视角做检测能更好避免截断问题。
+ 按照2D检测的pipeline，将2D中检测好的物体反投影回3D空间，需要预测深度、航向角等，引入了更多的误差
+ LiDAR的点云数据天然支持BEV视角的检测、识别，更方便融合多模态的数据，以及跟Prediction和Planning结合。

DETR3D 的2d 图像输入，标签是2d 框    
fpn输出shape？    
如何从2d到3d的转换？     
3D reference point 是训练学习得到的。     

基于输入数据，将BEV感知研究主要分为三个部分——BEV Camera、BEV LiDAR和BEV Fusion。

BEV Camera表示仅有视觉或以视觉为中心的算法，用于从多个周围摄像机进行三维目标检测或分割；  
BEV LiDAR描述了点云输入的检测或分割任务；  
BEV Fusion描述了来自多个传感器输入的融合机制，例如摄像头、激光雷达、全球导航卫星系统、里程计、高清地图、CAN总线等。

BEV Camrea中的代表之作是BEVFormer。BEVFormer 通过提取环视相机采集到的图像特征，并将提取的环视特征通过模型学习的方式转换到 BEV 空间（模型去学习如何将特征从图像坐标系转换到 BEV 坐标系），从而实现 3D 目标检测和地图分割任务，并取得了 SOTA 的效果。    
 
BEVFormer 的 Pipeline：      
1）Backbone + Neck （ResNet-101-DCN + FPN）提取环视图像的多尺度特征；    
2）论文提出的 Encoder 模块（包括 Temporal Self-Attention 模块和Spatial Cross-Attention 模块）完成环视图像特征向 BEV 特征的建模；     
3）类似 Deformable DETR 的 Decoder 模块完成 3D 目标检测的分类和定位任务；    
4）正负样本的定义（采用 Transformer 中常用的匈牙利匹配算法，Focal Loss + L1 Loss 的总损失和最小）；      
5）损失的计算（Focal Loss 分类损失 + L1 Loss 回归损失）；     
6）反向传播，更新网络模型参数；       

DETR3D  没有显式地引入BEV特征     transformer decoder  {  self-attention + crosss attention }       
BEVFormer 生成一个显式的BEV特征   时序特征融合(时序自注意力)temporal self-attention + 空间 spatial crosss attention     

+ 在 cross attention 中     
都是从 bev空间的3D query出发，得到参考点和采样点，再通过多相机的内外参投影到多视角2D图片上，和相应特征进行交互。  
但DETR3D的object queries是稀疏的，每个query代表一个可能的目标框   
而Bevformer的bev queries是稠密的，大小为H*W*C，H,W为设定的bev特征尺度，每个query代表一个网格（grid）的查询，这样可以得到稠密的bev特征。

+ 在 self-attention 中    
Bevformer 采用temporal self-attention 时序注意力，用前一帧和当前帧的bev特征进行交互，获取当前帧缺失的时序特征，用来解决当前帧目标遮挡或者不稳定的问题



从camera获取到数据之后，经过一层基础的训练学习，把它学习出一批基础的特征值。这些基础的特征值，一小部分用于全局性的任务处理，包括Free space以及场景的识别。
另外一部分会呈现多维的特征层，所有的识别任务就可以经过特征层来进行处理，包含了道路车道线、道路边界以及障碍物处理。 
通过一个主干网，以及两类neck，然后再处理8到9个这样的任务(Free space以及场景, 车道线、道路边界、动态障碍物的识别)

激光lidar层面也做了一些识别的工作，lidar处理方面现在行业里边，大家相对都会比较一致，会针对点云，做pointpillar特征的提取，做成一种伪2D的模式，再用图象识别的算法，把激光雷达里的一些障碍物和姿态给识别出来，




 
