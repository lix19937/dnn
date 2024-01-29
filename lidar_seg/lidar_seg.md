
ref centernet  

* 数据预处理
  `project_to_rv`    
  pcd(N, 3) or (N, 4) N代表点云点数目，3 or 4 代表  x, y, z, intensity    
  `(N, 3) -->  (C, H, W)`  (这里 N 因为点云会经过下采样和投影，一般都会有丢弃)， model的输入shape为 BCHW     
  using range view images(2d) from 3d pcd，好处在于很直观的将3D问题转换为比较成熟的2D问题，坏处就是丢失了点云数据的三维信息。    
  ```
  # prj to img coor index of col  row  
  c = ((yaw - self.fov_left) / self.horizon_fov * self.img_scale[2]).astype(np.int32)  # [N, ]
  r = (self.img_scale[1] - (pitch - self.fov_down) / self.fov * self.img_scale[1]).astype(np.int32)   # [N, ]

  dict_mat = dict()
  for idx in range(cloud_size):
      tmp_tuple = (r[idx], c[idx])
      if tmp_tuple not in dict_mat:
          dict_mat[tmp_tuple] = [idx] # just get intensity 
      else:
          dict_mat[tmp_tuple].append(idx)

  img_ori = np.zeros(self.img_scale, 'f')
  label_img = np.ones(self.img_scale[1:], np.uint8) * 10
  render_img = np.ones((*self.img_scale[1:], 3), np.uint8) * 0

  print("cloud_size {}, valid num {}".format(cloud_size, len(dict_mat)))
  for key, val in dict_mat.items():
      if self._in_roi(key): #  x, y  in range
          img_ori[0, key[0], key[1]] = ori_cloud[:, 0][val[-1]] # x
          img_ori[1, key[0], key[1]] = ori_cloud[:, 1][val[-1]] # y
          img_ori[2, key[0], key[1]] = ori_cloud[:, 2][val[-1]] # z
          img_ori[3, key[0], key[1]] = len(val)                 # >=1
          img_ori[4, key[0], key[1]] = depth[val[-1]]           # dist 
  ```    
  ![lidar_seg](https://github.com/lix19937/history/assets/38753233/765fab23-4b3e-40ab-87ae-dc4a8d9d5e46)
  

  |pcd bev| pcd2rv_img |
  |---|----|
  | ![1648609228 109629977_ML021_MW001_MT001_gt](https://github.com/lix19937/history/assets/38753233/fbcef6a1-3bbe-47c9-b4ee-4d06e9a76dbc)|![1648609228 109629977_ML021_MW001_MT001_gt](https://github.com/lix19937/history/assets/38753233/32bd7df7-d10b-4b47-877e-83048e0d830a) |    

* backbone   
  SalsaNext   2D图像的语义分割  
  https://arxiv.org/pdf/2003.03653.pdf  

* head   
  conv1x1 + softmax + argmax  得到 class +  prob   

* decode  
  主要就是 `backproject_from_rv`
  ```
  input      (B, C, H, W)    
  label      (B, H, W)             int32         --->    (N', 5)  fp32  
  label_prob (B, CLASS_NUM, H, W)  fp32  
  ```   
  ```
  for i in range(H):   
    for j in range(W):   
      if input[0, 3, i, j] < 1:
       continue
       
      cloud_out[k, 0] = input[0, 0, i, j] # x
      cloud_out[k, 1] = input[0, 1, i, j] # y
      cloud_out[k, 2] = input[0, 2, i, j] # z
      cloud_out[k, 3] = label[0, i, j]    # label
      cloud_out[k, 4] = label_prob[0, label[0, i, j], i, j] # prob
  ```
  
* 优化点    
  qat    
  cab to cba
