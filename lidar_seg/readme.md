conda  create -n lidar_seg     

conda activate lidar_seg    

cd seg/pytorch-quantization_v2.1.2    

python3 setup.py install --prefix=./site-packages    

conda deactivate      


# mount
sudo mount -t nfs 10.95.61.56:/data/softwares/seg_train_nfs    /Data/ljw/seg_train_nfs
cd  /Data/ljw/seg_train_nfs/seg/source

# nvidia@SSADL3816
conda activate CenterNet


# (CenterNet) nvidia@SSADL3816:/Data/ljw/seg_train_nfs/seg/source$   
10.95.61.56:/data/softwares/seg_train_nfs  1.8T  202G  1.6T  12% /Data/ljw/seg_train_nfs     

(centerpoint) root@SSADL3816:~/nv_centerpoint/CenterPoint#    
```
nvidia@SSADL3816:~$ conda env list
# conda environments:
#
alphanet_lane_fs         /home/nvidia/.conda/envs/alphanet_lane_fs
bevnet                   /home/nvidia/.conda/envs/bevnet
mmdet2                   /home/nvidia/.conda/envs/mmdet2
yolox_conda_env          /home/nvidia/.conda/envs/yolox_conda_env
base                  *  /usr/local/anaconda3
CenterNet                /usr/local/anaconda3/envs/CenterNet
```

```
nvidia@SSADL3816:~$ docker ps -a
CONTAINER ID   IMAGE                                                      COMMAND                  CREATED         STATUS                       PORTS                                                                                                  NAMES
6605e47dc790   dds_compile:v3.4.3                                         "bash"                   4 days ago      Up 4 days                    0.0.0.0:8801->22/tcp, :::8801->22/tcp                                                                  alg_data
7b4ba6f2b0eb   centerpoint:v2                                             "bash"                   6 days ago      Up 6 days  


nvidia@SSADL3816:~$ docker images 
REPOSITORY                                 TAG                       IMAGE ID       CREATED         SIZE
centerpoint                                v2                        fc0bbb55c8fe   6 days ago      42.5GB
```

```
10.95.61.56:/data/softwares/seg_train_nfs                                            1.8T  691G  1.1T  40% /Data/ljw/seg_train_nfs
```

//10.94.61.15/P_E2_DATA/数据标注/样例_障碍物/项目交付/LidarSEG/项目交付/luminar_seg  1.6P  1.6P  7.2T 100% /Data/luminar_seg_bk


##  train log    
https://github.com/lix19937/tensorrt-insight/issues/1   
