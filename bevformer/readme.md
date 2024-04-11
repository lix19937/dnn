
3D object detection and semantic map segmentation.     
https://www.zhihu.com/question/521842610/answer/2431585901   

https://github.com/fundamentalvision/BEVFormer    有3个分支       
|分支|说明|版本|    
|----|----|----|   
|BEVFormer v1.0 <br>[8240b652e5a3c22a14119f0f50d4e28ef8fc9441](https://github.com/fundamentalvision/BEVFormer/tree/v1.0)| an initial version of BEVFormer(the base version). It achieves a baseline result of **51.7%** NDS on nuScenes. | 对应 **BEVFormer** 论文 <br>BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers  |      
|BEVFormer v2.0 <br>[5a25c8b3b70956174b3c1876321967e265b2da42](https://github.com/fundamentalvision/BEVFormer/tree/v2.0)| require less GPU memory than the base version. Please pull this repo to obtain the latest codes. 支持fp16| 对应 **BEVFormer** 论文<br>BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers |                 
|BEVFormer master <br>[20923e66aa26a906ba8d21477c238567fa6285e9](https://github.com/fundamentalvision/BEVFormer/tree/20923e66aa26a906ba8d21477c238567fa6285e9)| 需要detectron2 | 对应 **BEVFormer v2** 论文实现 <br>BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision |       

### 使用v1.0-mini 生成训练需要的数据集格式
```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini --canbus ./data
```
可能出现 `ModuleNotFoundError`问题，可见https://github.com/open-mmlab/mmdetection3d/issues/2352#issuecomment-2044432207

### 单机train 运行     
![bevformerv2-2loss](https://github.com/lix19937/dnn-cookbook/assets/38753233/b9b12088-ad9f-4bb8-91b7-21a412b5241e)      

forward_mono_train/fcos3d_bbox_head        
`透视监督`是优化BEV模型的关键。在Bevformer V2中，我们通过一个辅助透视损失引入透视监督。 具体地说，在主干上构建透视3D检测头，以检测透视图中的目标对象。 我们采用FCOS3D-类似的检测头，它预测3D包围盒的中心位置、大小、方向和投影中心度。     

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
    lidar2img=[batch_size, cameras, 4, 4]
  # img_shape=[img_h, img_w] # 作为attribute 传入   
)
```

> 第 1 次 infer, use_prev_bev=0, prev_bev 使用默认值/随机值, 不参与fusion 运算, 得到 prev_bev_`1`    
> 第 k (k>1) 次 infer, use_prev_bev=1, prev_bev 使用prev_bev_`k-1`, 参与 fusion 运算, 得到 prev_bev_`k`     
> 条件分支的onnx 进行验证 https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#export-pytorch-example     

+ **1 对于 BEVFormerV2/obtain_history_bev 函数**        
  + 1.1 对历史帧进行 extract_feat（cnn 网络，仅使用img作为输入）   
  + 1.2 接着进行pts_bbox_head（BEVFormerHead/forward with only_bev=True）    
    + 1.2.1 进入 PerceptionTransformerV2/get_bev_features （实质是 PerceptionTransformerBEVEncoder/forward），返回得到 bev_embed（prev_bev）。
 
> 注意：在实际 infer 中，我们不进入obtain_history_bev，直接传入历史 prev_bev 集合，避免再计算。    

+ **2 对于 BEVFormerV2/extract_feat 函数**    
  + 2.1 对当前帧进行 extract_feat（cnn 网络，仅使用img作为输入），返回得到 img_feats 注意 len(img_feats) 由 `_num_mono_levels_` 控制。
  + 2.2 随后 img_feats 还会被 slice操作 `img_feats = img_feats[:self.num_levels]`，因此 如果 `_num_levels_ < _num_mono_levels_` ，则 extract_feat 存在冗余计算。
    
> 注意：可以单独导出 cnn 网络，分析计算图进行优化。          

+ **3 对于 BEVFormerV2/simple_test_pts 函数**         
使用步骤2 得到的 img_feats 以及历史帧的 **prev_bev** 进行 simple_test_pts(img_feats, img_metas, prev_bev)    
    + 3.1 pts_bbox_head（BEVFormerHead/forward with only_bev=False, prev_bev!=None）
      + 3.1.1 PerceptionTransformerV2/forward，返回 `bev_embed, inter_states, init_reference_points_out, inter_references_out`。        
        + 3.1.1.1 PerceptionTransformerV2/get_bev_features（实质是encoder，PerceptionTransformerBEVEncoder/forward），得到 bev_embed（prev_bev）。
        + 3.1.1.2 当前 bev_embed 与历史 prev_bev 进行**融合**，计算query, query_pos, reference_points, bev_embed      
          ```py
          bev_embed_m = [x.reshape(x.shape[0], bev_h, bev_w, x.shape[-1]).permute(0, 3, 1, 2).contiguous() for x in prev_bev]
          bev_embed = self.fusion(bev_embed_m) # ResNetFusion   bev_embed_m [(1, embedding_dim, bev_h, bev_w), (1, embedding_dim, bev_h, bev_w)]
  
          bs = mlvl_feats[0].size(0)
          query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
          query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
          query = query.unsqueeze(0).expand(bs, -1, -1)
          reference_points = self.reference_points(query_pos)
          reference_points = reference_points.sigmoid()
          init_reference_out = reference_points
  
          query = query.permute(1, 0, 2)
          query_pos = query_pos.permute(1, 0, 2)
          bev_embed = bev_embed.permute(1, 0, 2)
          ```
        + 3.1.1.3 PerceptionTransformerV2/decoder 得到 
          ```py
          inter_states, inter_references = self.decoder(
              query=query,
              key=None,
              value=bev_embed,
              query_pos=query_pos,
              reference_points=reference_points,
              reg_branches=reg_branches,
              cls_branches=cls_branches,
              spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
              level_start_index=torch.tensor([0], device=query.device),
              **kwargs)
          ```
        + 3.1.1.4 计算 outputs_coord with reg_branches    
        + 3.1.1.5 计算 outputs_class with cls_branches      
          
  > 注意：这一步最终返回
  > outs = {     
  > &emsp;&emsp;&emsp;'bev_embed': bev_embed,&emsp;&emsp;&emsp;   # [bev_h*bev_w, batch_size, embedding_dim]    
  > &emsp;&emsp;&emsp;'all_cls_scores': outputs_classes,&ensp;    # [num_camera, batch_size, num_query, num_classes]    
  > &emsp;&emsp;&emsp;'all_bbox_preds': outputs_coords            # [num_camera, batch_size, num_query, code_size]    
  >         }     
  > 包含了encoder + decoder，计算量巨大，重点优化。
  > img_metas 未参与计算
  
    + 3.2 pts_bbox_head.get_bboxes（BEVFormerHead/get_bboxes），基于预测点生成bbox（LiDARInstance3DBoxes坐标系）
      + 3.2.1 bbox_coder.decode（NMSFreeCoder/decode）会进行topK的过滤（k取自于 max_num，一般比num_query 要小）   
        ```py   
        all_bbox_preds torch.Size([6, 1, 900, 10])
        cls_scores torch.Size([1, 900, 10]) # num_query = 900 
        bbox_preds torch.Size([1, 900, 10])
        score_threshold None, max_num 300
   
        bboxes.shape torch.Size([300, 9]), code_size 9
        new_prev_bev torch.Size([40000, 1, 256])
        boxes_3d  torch.Size([300, 9]) # max_num = 300  
        scores_3d torch.Size([300])
        labels_3d torch.Size([300])
        ```    
      + 3.2.2 执行 `img_metas[i]['box_type_3d'](bboxes, code_size)`，此处后续可以固化，直接选取mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes(bboxes, code_size)              

> 注意：infer 阶段没有 fcos3d_bbox_head 分支   
    
bevformer-master    
+ pth2onnx

  
