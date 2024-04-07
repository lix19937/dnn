## 目录   
- [detr](#detr )     
- [detr3d](#detr3d )     
- [Ref](#Ref )     
  
# detr   
> https://arxiv.org/pdf/2005.12872.pdf  Facebook 提出的       
![detr](https://github.com/lix19937/pytorch-cookbook/assets/38753233/10aca5e6-a62e-478d-b4bd-16e1a79f1be5)
![detr-detailed](https://github.com/lix19937/pytorch-cookbook/assets/38753233/1f3a29f1-62bf-404c-b354-b42dea11caff)   

a `CNN backbone` to extract a compact feature representation, an `encoder-decoder transformer`, and a simple `feed forward network (FFN)` that makes the final detection prediction.

---------------------------   
# detr3d      
> https://arxiv.org/abs/2110.06922    MIT wangyue 等提出的    
https://github.com/wangyueft/detr3d       
多摄像头检测目标，输出为BEV视角下的目标框, 无点云, 仅以 cnn (csp+darknet53, yolox used) +  FPN  + transformer
https://github.com/WangYueFt/detr3d/blob/main/projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py    
DETR3D 主要解决自动驾驶中的三维物体检测问题，还可以应用于室内机器人、监控摄像头的物体检测。 DETR3D`不依赖视觉深度预测`，直接在3D中进行检测，其次，DETR3D算法针对多个相机作为整体进行检测，`无需后处理(如NMS去除冗余三维检测框)、相机间的跟踪、相机融合`。    
> ![flow](https://github.com/lix19937/pytorch-cookbook/assets/38753233/3525dd0b-26c9-4e42-99eb-6cd62575d4b9)

* 标签
  |名称|类型|其他 |    
  |---|---|---|   
  |- |- |- |

* 输入       
  |名称|shape|类型|其他 |    
  |---|---|---|---|     
  |image| (6,3,288,480)| fp32 |车体环视6相机的6张图片 c=3,h=288,w=480 <br><br>顺序如下<br> F120 <br> B70 <br> ARO233_BACK_RIGHT<br>ARO233_FRONT_RIGHT <br> ARO233_FRONT_LEFT <br> ARO233_BACK_LEFT |   
  |vehicle2img| (6,4,4) or (6,3,4)| fp32 | 相机参数矩阵[3, 4]可以padding一行[4, 4], 凑成方阵, 共6个<br>这里我们使用(6,4,4)|     

* 输出
  |名称|shape|类型|其他 |    
  |---|---|---|---|     
  |bbox| (1,512,8)| fp32 |512是query num, 8是每一个bbox的属性组成 (bbox的坐标xyz+长宽高+航向角+速度)   |   
  |score| (1,512,12) | fp32 |512是query num, 12是检测类别数 |
  
  ![out](https://github.com/lix19937/pytorch-cookbook/assets/38753233/56209a1f-cfe4-4de4-ac76-f06e528e7f57)
  
---------------------------------

* 数据预处理  
  RGB色彩空间下
  ```cpp  
  (float(x)-mean)/std
  ```

* backbone + neck     
  输入车载环视的6（NC=6）张图片, 每张图片通过ResNet或者VoVNet等骨干网络提取特征；再通过FPN得到4个不同尺度特征图      
  |模块| 作用|    
  |---|----|    
  |csp+darknet53|Feature Learning, 输出给FPN (特征金字塔) |       
  |[FPN](fpn/fpn.md)| Multi-scale features provide rich information to recognize objects of different sizes<br>4个不同尺度(H,W)特征图<br>(NC, CH, 72,184)<br>(NC, CH, 36,92)<br>(NC, CH, 18,46)<br>(NC, CH,  9,23)<br> 这里我们使用https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/necks/pafpn.py|         

* transformer decoder       
`特征层面实现2D到3D的转换`          
用于从相机输入检测对象的现有方法通常采用`自下而上`的方法，其预测每个图像的密集边界框集合、过滤图像之间的冗余框，并且在后处理步骤中聚合跨相机的预测。    
这种模式有两个缺点：
    + 密集边界框预测需要精确的深度感知，而深度感知本身就是一种具有挑战性的问题；   
    + 基于NMS的冗余删除和合并是不可并行的引入大量推理开销的操作。这里使用`自上而下`的方法来解决这些问题。
 
  注意：       
  decoder的输入是经过fpn后的特征图，在特征层面实现2D到3D的转换，避免深度估计带来的误差，同时可以避免NMS等耗时的后处理操作。      
  ![model-detr3d](https://github.com/lix19937/pytorch-cookbook/assets/38753233/7b256cca-adfe-4d1f-8243-539eb5020d28)    
  decoder的部分输入紧连cross attention（注意不是紧连MHA），见[cross attention](https://github.com/lix19937/tensorrt-insight/tree/main/plugin/detr3d/decoder/cross_attention.md) 

-----------------  
+ 预先设置query nums (seq_lens) 600/or 900， 取512个object query（1个query 预测一个目标，bbox的3维中心点坐标3D reference point），每个query是256 维的 embedding。    
所有的 object query由一个全连接网络预测出在BEV空间中的3D reference point坐标(x, y, z)，坐标经过sigmoid函数归一化后表示在空间中的相对位置。

+ 在每层layer之中，所有的object query之间做self-attention来相互交互获取全局信息并避免多个query收敛到同个物体。   
object query再和图像特征之间做cross-attention：将每个query对应的3D reference point通过相机的内参外参投影到特征图图片坐标系（将reference point转为齐次坐标，通过相机参数矩阵转为2D中心点，特别注意的是，每个相机标定的参数矩阵都是不一样的，需要成对应的参数矩阵）
利用线性插值来采样对应的multi-scale image features，如果投影坐标落在图片范围之外就补零，之后再用sampled image features去更新object queries。  

+ 经过attention更新后的object query通过两个MLP网络来分别预测对应物体的class和bounding box的参数。为了让网络更好的学习，我们每次都预测bounding box的中心坐标相对于reference points的offset (delta_x, delta_y, delta_z) 来更新reference points的坐标。

+ 每层更新的object queries和reference points作为下一层decoder layer的输入，再次进行计算更新，总共迭代6次。    

-------------------------

+ 如何将环视图像转化为BEV？   
在DETR3D、BEVFormer中，是通过reference points和相机参数 的物理意义进行投影来获取图像features，这样的优点在于计算量较小，通过FPN的mutli-scale结构和deformable detr的learned offset，即使只有一个或几个reference points也可以得到足够的感受野信息。缺点在于BEV的同个polar ray上的reference point通过投影采样到的图像特征都是一样的，图像缺少了深度信息，网络需要在后续特征聚合的时候去判别采样到的信息和当前位置的reference points是否match。   
在BEVDet里，转化过程follow了lift-splat-shoot的方法，也就是对image feature map的每个位置预测一个depth distribution，再将feature的值乘以深度概率lift到BEV下。这么做需要很大的计算量和显存，由于没有真实的深度标签，所以实际预测的是一个没有确切物理意义的概率。而且图片中相当一部分内容是不含有物体的，将全部feature参与计算可能略显冗余。

+ 如何选择BEV的表现形式？   
在DETR3D里，我们并没有完整显式地表示出了整个BEV，而且由sparse的object query来进行表示。最显著的好处就是节省了内存和计算量。而在BEVDet和BEVFormer里，他们生成了一个dense的BEV feature，虽然增加了显存，不过一来更容易去做BEV space下的data augmentation，二来像BEVDet一样可以另外增加对BEV features 的encoding，三来可以适应于各种3D detection head（BEVDet用了centerpoint，BEVFormer用了deformable detr）。

---------------------------

* head    
输出通过两个分支，`回归bbox信息`和`分类目标类别`
  
* loss     
采用detr 的set-to-set计算方式，对预测出的(回归, 分类)结果和GT的(回归, 分类)结果进行匹配。损失函数部分保持和 detr 一致。回归损失采用L1，分类损失使用focal loss。      
-------  

* **优化点**  
  * backbone focus结构替换
  * fpn的maxpool后融合至插件中
  * backbone + fpn 进行ptq 后量化   
  * **transformer decoder**结构通过手写插件替换trt native (myelin)实现，具体内容可见[svt](https://github.com/lix19937/tensorrt-insight/tree/main/plugin/svt)               
    注意：这里的token 只有一个 因此没有kv_cache    
  * head 模块也融入到自定义插件中     
    
  如果是[gpt](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)类生成模型,输入的token往往很多,因此需要kv_cache,对于时序detr3d则需要考虑         
       

## Ref      
https://zhuanlan.zhihu.com/p/587380480   
https://zhuanlan.zhihu.com/p/499795161   

 
