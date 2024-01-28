
#  
export CUDA_VISIBLE_DEVICES="2,3,4,5,6,7" 	
export NCCL_P2P_LEVEL=NVL#  include calib  from strach
python -m torch.distributed.launch --nproc_per_node=6  \
main.py lidarv5-tmp-local-shuffle-0527 --user_spec --batch_size 16 --gpus 0,1,2,3,4,5 \
--num_workers 112  --down_ratio 4  --val_intervals 5   \
--num_epochs 600  --aug_lidar 0.8  --qdq  --exec_calib  --save_all  \
--fp32_ckpt_file /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq_v4-full/model_1129.pth \
--exp_id test 

 