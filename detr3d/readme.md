  
# detr   
> https://arxiv.org/pdf/2005.12872.pdf      
![detr](https://github.com/lix19937/pytorch-cookbook/assets/38753233/10aca5e6-a62e-478d-b4bd-16e1a79f1be5)
![detr-detailed](https://github.com/lix19937/pytorch-cookbook/assets/38753233/1f3a29f1-62bf-404c-b354-b42dea11caff)   

a `CNN backbone` to extract a compact feature representation, an `encoder-decoder transformer`, and a simple `feed forward network (FFN)` that makes the final detection prediction.

---------------------------   
# detr3d      
> https://arxiv.org/abs/2110.06922  
https://github.com/wangyueft/detr3d       
多摄像头检测目标，输出为BEV视角下的目标框, 无点云, 仅以 cnn (csp+darknet53, yolox used) +  FPN  + transformer
https://github.com/WangYueFt/detr3d/blob/main/projects/configs/detr3d/detr3d_vovnet_gridmask_det_final_trainval_cbgs.py    
DETR3D 主要解决自动驾驶中的三维物体检测问题，还可以应用于室内机器人、监控摄像头的物体检测。 DETR3D`不依赖视觉深度预测`，直接在3D中进行检测，其次，DETR3D算法针对多个相机作为整体进行检测，`无需后处理(如NMS去除冗余三维检测框)、相机间的跟踪、相机融合`。    
> ![flow](https://github.com/lix19937/pytorch-cookbook/assets/38753233/3525dd0b-26c9-4e42-99eb-6cd62575d4b9)

* 输入       
  |名称|shape|类型|其他 |    
  |---|---|---|---|     
  |image| (6,3,288,480)| fp32 |车体环视6相机的6张图片 c=3,h=288,w=480 <br><br>顺序<br> F120 <br> B70 <br> ARO233_BACK_RIGHT<br>ARO233_FRONT_RIGHT <br> ARO233_FRONT_LEFT <br> ARO233_BACK_LEFT |   
  |vehicle2img| (6,4,4) or (6,3,4)| fp32 | 相机参数矩阵[3, 4]可以padding一行[4, 4], 凑成方阵, 共6个<br>这里使用(6,4,4)|     

* 输出
  |名称|shape|类型|其他 |    
  |---|---|---|---|     
  |bbox| (1,512,8)| fp32 |512是query num, 8是每一个bbox的属性组成 (bbox的坐标+长宽高+航向角+速度)   |   
  |score| (1,512,12) | fp32 |512是query num, 12是检测类别数 |
  
  ![out](https://github.com/lix19937/pytorch-cookbook/assets/38753233/56209a1f-cfe4-4de4-ac76-f06e528e7f57)
  
---------------------------------

* 数据预处理  
  RGB色彩空间下
  ```cpp  
  (float(x)-mean)/std
  ```

* backbone + neck     
  输入车载环视的6张图片, 每张图片通过ResNet或者VoVNet等骨干网络提取特征；再通过FPN得到4个不同尺度特征图      
  |模块| 作用|    
  |---|----|    
  |csp+darknet53|Feature Learning, 输出给FPN (特征金字塔) |       
  |[FPN](fpn.md)| Multi-scale features provide rich information to recognize objects of different sizes<br>4个不同尺度(H,W)特征图<br>(72,184)<br>(36,92)<br>(18,46)<br>(9,23)|         

* transformer decoder
  `特征层面实现2D到3D的转换`          
用于从相机输入检测对象的现有方法通常采用`自下而上`的方法，其预测每个图像的密集边界框集合、过滤图像之间的冗余框，并且在后处理步骤中聚合跨相机的预测。    
这种模式有两个缺点：密集边界框预测需要精确的深度感知，而深度感知本身就是一种具有挑战性的问题；
基于NMS的冗余删除和合并是不可并行的引入大量推理开销的操作。这里使用`自上而下`的方法来解决这些问题。   
decoder的输入是经过fpn后的特征图，在特征层面实现2D到3D的转换，避免深度估计带来的误差，同时可以避免NMS等耗时的后处理操作。      
![model-detr3d](https://github.com/lix19937/pytorch-cookbook/assets/38753233/7b256cca-adfe-4d1f-8243-539eb5020d28)    

* head    
输出通过两个分支，`回归bbox信息`和`分类目标类别`
  
* loss: 采用DETR的set-to-set计算方式，对预测出的(回归, 分类)结果和GT的(回归, 分类)结果进行匹配。损失函数部分保持和DETR一致。回归损失采用L1，分类损失使用focal loss      
-------  

* **优化点**  
  * backbone focus结构替换, maxpool后融合    
  * **transformer decoder**结构通过手写插件替换trt native实现          
    注意:这里的token 只有一个 因此没有kv_cache    
  * head 模块也融入到自定义插件中
    
  如果是[gpt](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)类生成模型,输入的token往往很多,因此需要kv_cache,对于时序detr3d则需要考虑         
       


## REF  
https://zhuanlan.zhihu.com/p/587380480   
https://zhuanlan.zhihu.com/p/499795161   

 
