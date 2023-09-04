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
多摄像头检测目标，输出为BEV视角下的目标框, no point clouds   cnn (csp+darknet53, yolox used) +  FPN  + transformer

* 输入       
  |名称|shape|类型|其他 |    
  |---|---|---|---|     
  |image| (6,3,288,480)| int32 |   |   
  |vehicle2img| (6,4,4)| fp32 |   |     


* 数据预处理  


* backbone   
  csp+darknet53 + FPN   
  
  csp+darknet53: Feature Learning   
  FPN: multi-scale features provide rich information to recognize objects of different sizes.    

* head   
transformer decoder


* 优化点   
  transformer decoder 结构  
  注意:这里的token 只有一个 因此没有kv_cache    
  
  如果是 gpt 类生成模型,输入的token往往很多,因此需要kv_cache    
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py  

 
