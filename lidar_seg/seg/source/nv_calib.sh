
export CUDA_VISIBLE_DEVICES="4,5,6,7" 	

python3  ./nv_calib.py nv_calib  --qdq  \
--fp32_ckpt_file /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq/model_best.pth \
--down_ratio 4 --gpus 4   
