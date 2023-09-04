##  svod  

* ref detr3d    
多摄像头检测目标，输出为BEV视角下的目标框,   cnn (csp+darknet53, yolox used)   + transformer


* 输入     
image(6,3,288,480) int32, 	 vehicle2img(6,4,4) fp32 

* 数据预处理  


* backbone   
csp+darknet53

* head   
transformer decoder


* 优化点   
  transformer decoder 结构  
  注意:这里的token 只有一个 因此没有kv_cache    
  
  如果是 gpt 类生成模型,输入的token往往很多,因此需要kv_cache    
  https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py  

 
