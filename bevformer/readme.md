
3D object detection and semantic map segmentation.     
https://www.zhihu.com/question/521842610/answer/2431585901   

https://github.com/fundamentalvision/BEVFormer    有3个分支       
|分支|说明|版本|    
|----|----|----|   
|BEVFormer v1.0| an initial version of BEVFormer(the base version). It achieves a baseline result of **51.7%** NDS on nuScenes. | 对应 **BEVFormer** 论文 <br>BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers  |      
|BEVFormer v2.0| require less GPU memory than the base version. Please pull this repo to obtain the latest codes. 支持fp16| 对应 **BEVFormer** 论文  <br>BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers |                 
|BEVFormer master| 需要detectron2 | 对应 **BEVFormer v2** 论文实现 <br>BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision |       

### 使用v1.0-mini 生成训练需要的数据集格式
```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```
可能出现 `ModuleNotFoundError`问题，可见https://github.com/open-mmlab/mmdetection3d/issues/2352#issuecomment-2044432207

### 单机test 运行   
+ configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py    

```shell
./tools/dist_test.sh  ./projects/configs/bevformerv2/bevformerv2-r50-t1-base-24ep.py ./ckpts/BEVFormerV2/bevformerv2-r50-t1-base/epoch_24.pth
```

img_metas 数据dump如下       
```
dict_keys(['img_metas', 'img', 'ego2global_translation', 'ego2global_rotation', 'lidar2ego_translation', 'lidar2ego_rotation', 'timestamp'])

img_metas ==== <class 'list'>   1   <class 'mmcv.parallel.data_container.DataContainer'> 
DataContainer([[
{0: {
'filename': [
'./data/nuscenes/samples/CAM_FRONT/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg', 
'./data/nuscenes/samples/CAM_FRONT_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_RIGHT__1533151603520482.jpg', 
'./data/nuscenes/samples/CAM_FRONT_LEFT/n008-2018-08-01-15-16-36-0400__CAM_FRONT_LEFT__1533151603504799.jpg',
'./data/nuscenes/samples/CAM_BACK/n008-2018-08-01-15-16-36-0400__CAM_BACK__1533151603537558.jpg', 
'./data/nuscenes/samples/CAM_BACK_LEFT/n008-2018-08-01-15-16-36-0400__CAM_BACK_LEFT__1533151603547405.jpg', 
'./data/nuscenes/samples/CAM_BACK_RIGHT/n008-2018-08-01-15-16-36-0400__CAM_BACK_RIGHT__1533151603528113.jpg'
], 
'ori_shape': [(640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3)], 
'img_shape': [(640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3)], 
'lidar2img': shape(6, 4, 4) 
'lidar2cam': shape(6, 4, 4)
'cam2img':   shape(6, 4, 4)
'pad_shape': [(640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3), (640, 1600, 3)], 
'scale_factor': 1.0, 
'flip': False, 
'pcd_horizontal_flip': False, 
'pcd_vertical_flip': False,
'box_mode_3d': <Box3DMode.LIDAR: 0>,
'box_type_3d': <class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'>,
'img_norm_cfg': {'mean': array([103.53 , 116.28 , 123.675], dtype=float32), 'std': array([1., 1., 1.], dtype=float32), 'to_rgb': False}, 'sample_idx': '3e8750f331d7499e9b5123e9eb70f2e2', 'pcd_scale_factor': 1.0, 
                               'pts_filename': './data/nuscenes/samples/LIDAR_TOP/n008-2018-08-01-15-16-36-0400__LIDAR_TOP__1533151603547590.pcd.bin', 'scene_token': 'fcbccedd61424f1b85dcbf8f897f9754', 
                               'timestamp': 1533151603.54759,'lidaradj2lidarcurr': None}
}
]])

img ==== torch.Size([1, 1, 6, 3, 640, 1600])

ego2global_translation ==== <class 'list'>   1   <class 'list'> 
 [tensor([600.1202],  dtype=torch.float64),
  tensor([1647.4908], dtype=torch.float64),
  tensor([0.],        dtype=torch.float64)]

ego2global_rotation ==== <class 'list'>   1   <class 'list'> 
 [tensor([-0.9687], dtype=torch.float64), 
 tensor([-0.0040],  dtype=torch.float64), 
 tensor([-0.0077],  dtype=torch.float64), 
 tensor([0.2482],   dtype=torch.float64)]

lidar2ego_translation ==== <class 'list'>   1   <class 'list'> 
 [tensor([0.9858], dtype=torch.float64), 
  tensor([0.],     dtype=torch.float64), 
  tensor([1.8402], dtype=torch.float64)]

lidar2ego_rotation ==== <class 'list'>   1   <class 'list'> 
 [tensor([0.7067],  dtype=torch.float64), 
  tensor([-0.0153], dtype=torch.float64), 
  tensor([0.0174],  dtype=torch.float64), 
  tensor([-0.7071], dtype=torch.float64)]

timestamp ==== <class 'list'>   1   <class 'torch.Tensor'> 
 tensor([1.5332e+09], dtype=torch.float64)
```

```
model()/forward -> forward_test -> simple_test -> extract_feat(img) (@cnn) -> extract_img_feat      
                                                      | -> simple_test_pts (@transformer)  
```

在 infer阶段， extract_feat 不使用 img_metas 数据     
```
input_shapes = dict(
    image   =[batch_size, cameras, 3, img_h, img_w],  # 
    prev_bev=[bev_h*bev_w, batch_size, embed_dim],    # [40000, 1, 256]
    use_prev_bev=[1],
    lidar2img=[batch_size, cameras, 4, 4],
)
```
第 1 次 infer, use_prev_bev=0, prev_bev 使用默认值/随机值, 不参与运算, 得到 prev_bev_`1`    
第 k (k>1) 次 infer, use_prev_bev=1, prev_bev 使用prev_bev_`k-1`, 参与运算, 得到 prev_bev_`k`     

+ 对于 BEVFormerV2/obtain_history_bev 函数，对历史帧进行 extract_feat（cnn 网络，仅使用img作为输入），接着进行（BEVFormerHead forward with only_bev=True）pts_bbox_head， 即进入（PerceptionTransformerV2）的 get_bev_features （实质是 PerceptionTransformerBEVEncoder 的 forward），返回得到 bev_embed（prev_bev）。   
在实际 infer 中，我们直接传入历史 prev_bev 集合，避免再计算。

+ 对于 BEVFormerV2/extract_feat 函数，对当前帧进行 extract_feat（cnn 网络，仅使用img作为输入），返回得到 img_feats 注意 len() 由 `_num_mono_levels_`控制，而最终
img_feats 还会被 slice操作 ` img_feats = img_feats[:self.num_levels]`，因此 如果 ` _num_levels_ < _num_mono_levels_ ` ，则 extract_feat 存在冗余计算。
可以单独导出，分析计算图。         

+ 对于 BEVFormerV2/simple_test_pts 函数，对上一步骤得到的 img_feats 以及 历史帧的 prev_bev 进行 simple_test_pts(img_feats, img_metas, prev_bev)， simple_test_pts包含两个步骤：（BEVFormerHead）pts_bbox_head 和 （BEVFormerHead）pts_bbox_head.get_bboxes

bevformer-master    
+ pth2onnx

  
