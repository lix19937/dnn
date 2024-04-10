
3D object detection and semantic map segmentation.     
https://www.zhihu.com/question/521842610/answer/2431585901   

https://github.com/fundamentalvision/BEVFormer    有3个分枝    
|分枝|说明|版本|    
|----|----|----|   
|BEVFormer v1.0| an initial version of BEVFormer(the base version). It achieves a baseline result of **51.7%** NDS on nuScenes. | 对应BEVFormer 论文 <br>BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers  |      
|BEVFormer v2.0| require less GPU memory than the base version. Please pull this repo to obtain the latest codes. 支持fp16| 对应BEVFormer 论文  <br>BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers |                 
|BEVFormer master| 需要detectron2 | 对应 BEVFormer v2 论文实现 <br>BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision |       

### 使用v1.0-mini 生成训练需要的数据集格式
```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```
可能出现 `ModuleNotFoundError`问题，可见https://github.com/open-mmlab/mmdetection3d/issues/2352#issuecomment-2044432207

### 单机test 运行   
```shell
./tools/dist_test.sh  ./projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py ./ckpts/BEVFormerV2/bevformerv2-r50-t1-base/epoch_24.pth
```

bevformer-master    
+ pth2onnx

  
