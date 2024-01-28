export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 	

python -m torch.distributed.launch --nproc_per_node=8  \
main.py lidarv3  --batch_size 16 --gpus 0,1,2,3,4,5,6,7 \
--num_workers 112  --down_ratio 4  --val_intervals 5  \
--num_epochs 400  --aug_lidar 0.8   \
--load_model  /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq/model_1129.pth
