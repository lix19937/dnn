    
    # python saic_od_eva.py --dt_root /Data/lidar_data/seg-2.0w/aligned_seg/ \
    # --gt_root /home/igs/Desktop/eval_test_data/bin/labels/pull/label/ --Is_class_D 1

    #Is_class_D 0:without class  1:withclass


export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 	

# /../exp/lidarv2/default/model_last.pth, epoch:261


## calib eval
# python infer_lidar_od.py  lidarv5-tmp-local-shuffle-0403-eval-ptq --user_spec --qdq --down_ratio 4 --gpus 8  --batch_size 64 \
#  --load_model /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq/model_1129_calib_0.pth

## best eval
python infer_lidar_od.py lidarv5-tmp-local-shuffle-0527-eval-best --user_spec --qdq --down_ratio 4 --gpus 8  --batch_size 64 \
 --load_model /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0527/test/model_best.pth  

## last eval
python infer_lidar_od.py lidarv5-tmp-local-shuffle-0527-eval-last --user_spec --qdq --down_ratio 4 --gpus 8  --batch_size 64 \
 --load_model /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0527/test/model_last.pth


# python infer_lidar_od.py  lidarv5-tmp-local-shuffle-0403-eval-ptq-entropy --user_spec --qdq --down_ratio 4 --gpus 8  --batch_size 64 \
#  --load_model /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq_ddp/model_1129_calib_0_entropy_.pth  

