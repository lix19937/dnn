
#  
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" 	
export NCCL_P2P_LEVEL=NVL

# resume
python -m torch.distributed.launch --nproc_per_node=6  \
main.py lidarv5-tmp-local-shuffle-0527 --user_spec --batch_size 16 --gpus 0,1,2,3,4,5 \
--num_workers 112  --down_ratio 4  --val_intervals 5  --resume \
--num_epochs 600  --aug_lidar 0.8  --qdq    --save_all  \
--fp32_ckpt_file /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0527/test/model_last.pth \
--exp_id test 


# finetune
# python -m torch.distributed.launch --nproc_per_node=8  \
# main.py lidarv5-tmp-local-shuffle-0313-finetune  --user_spec --batch_size 16 --gpus 0,1,2,3,4,5,6,7 \
# --num_workers 112  --down_ratio 4  --val_intervals 5  --lr 1.e-6 \
# --num_epochs 400  --aug_lidar 0.8  --qdq  --save_all  \
# --fp32_ckpt_file /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0315/test/model_best.pth \
# --exp_id test  

# /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq/model_1129_calib_0.pth
# /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0302-resume/test/model_last.pth


