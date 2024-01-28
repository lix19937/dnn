
#  
export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" 	
export NCCL_P2P_LEVEL=NVL

python -m torch.distributed.launch --nproc_per_node=8  \
nv_calib_ddp.py  vvv --gpus 0,1,2,3,4,5,6,7 \
--num_workers 112  --down_ratio 4  --qdq  --exec_calib   \
--fp32_ckpt_file /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq_ddp/model_1129.pth 


 