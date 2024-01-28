    
    # python saic_od_eva.py --dt_root /Data/lidar_data/seg-2.0w/aligned_seg/ \
    # --gt_root /home/igs/Desktop/eval_test_data/bin/labels/pull/label/ --Is_class_D 1

    #Is_class_D 0:without class  1:withclass
	
#python infer_lidar_od.py lidar --down_ratio 4 --gpus 4 \
# --load_model /Data/ljw/seg_train_nfs/seg/exp/lidar/default/model_best.pth

export CUDA_VISIBLE_DEVICES="4,5,6,7" 	

# python infer_lidar_od.py lidarv2  --down_ratio 4 --gpus 4  --batch_size 64 \
# --load_model /Data/ljw/seg_train_nfs/seg/exp/lidarv2/default/model_best.pth

#  python infer_lidar_od.py lidarv2  --down_ratio 4 --gpus 4  --batch_size 64 \
#  --load_model /Data/ljw/seg_train_nfs/seg/exp/lidarv2/default/fp32_nq/model_best.pth


#

python infer_lidar_od.py fp32_nq-V2   --down_ratio 4 --gpus 4  --batch_size 64 \
--load_model /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq/model_1129.pth


 