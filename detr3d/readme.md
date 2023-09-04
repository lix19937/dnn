--------------------------   
detr https://arxiv.org/pdf/2005.12872.pdf
![detr](https://github.com/lix19937/pytorch-cookbook/assets/38753233/10aca5e6-a62e-478d-b4bd-16e1a79f1be5)
![detr-detailed](https://github.com/lix19937/pytorch-cookbook/assets/38753233/1f3a29f1-62bf-404c-b354-b42dea11caff)   

a `CNN backbone` to extract a compact feature representation, an `encoder-decoder transformer`, and a simple `feed forward network (FFN)` that makes the final detection prediction.

---------------------------   
detr3d   
![flow](https://github.com/lix19937/pytorch-cookbook/assets/38753233/3525dd0b-26c9-4e42-99eb-6cd62575d4b9)    


* ref detr3d  
https://arxiv.org/abs/2110.06922  
https://github.com/wangyueft/detr3d       
多摄像头检测目标，输出为BEV视角下的目标框, no point clouds, cnn (csp+darknet53, yolox used) +  FPN  + transformer

* 输入       
  |名称|shape|类型|其他 |    
  |---|---|---|---|     
  |image| (6,3,288,480)| int32 |车体环视6相机的6张图片   |   
  |vehicle2img| (6,4,4) or (6,3,4)| fp32 | 相机参数矩阵[3, 4]可以padding一行[4, 4] , 凑成方阵, 共6个|     


* 数据预处理  


* backbone   
  |模块| 作用|    
  |---|----|    
  |csp+darknet53|Feature Learning |       
  |FPN| multi-scale features provide rich information to recognize objects of different sizes|         

* transformer decoder
  特征层面实现2D到3D的转换   
  用于从相机输入检测对象的现有方法通常采用自下而上的方法，其预测每个图像的密集边界框集合、过滤图像之间的冗余框，并且在后处理步骤中聚合跨相机的预测。这种模式有两个缺点：密集边界框预测需要精确的深度感知，而深度感知本身就是一种
  具有挑战性的问题；基于NMS的冗余删除和合并是不可并行的引入大量推理开销的操作。这里使用`自上而下`的方法来解决这些问题   

* head    

* 优化点   
  transformer decoder 结构  
  注意:这里的token 只有一个 因此没有kv_cache    
  
  如果是 gpt 类生成模型,输入的token往往很多,因此需要kv_cache    
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py  

 
