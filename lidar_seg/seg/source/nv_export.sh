
export CUDA_VISIBLE_DEVICES="0，1" 	

# python3   ./nv_export.py nv_export  --qdq  \
# --ptq_pth_file /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq/model_1129_calib_0.pth \
# --down_ratio 4 --gpus 4   #  > log.txt 2>&1



# python3   ./nv_export.py nv_export  --qdq  \
# --ptq_pth_file /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0403-eval-best/default/eval_2023-04-18-09-31/model_best.pth \
# --down_ratio 4 --gpus 4   


python3   ./nv_export.py nv_export  --qdq  \
--ptq_pth_file /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0403/test/model_last.pth \
--down_ratio 4 --gpus 4   


python3   ./nv_export.py nv_export  --qdq  \
--ptq_pth_file /Data/ljw/seg_train_nfs/seg/exp/lidarv5-tmp-local-shuffle-0403/test/model_best.pth \
--down_ratio 4 --gpus 4  

# python3   ./nv_export.py nv_export  --qdq  \
# --ptq_pth_file /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq_ddp/model_1129_calib_2.pth \
# --down_ratio 4 --gpus 4   

# python3   ./nv_export.py nv_export  --qdq  \
# --ptq_pth_file /Data/ljw/seg_train_nfs/seg/exp/full_trained/default/fp32_nq_ddp/model_1129_calib_3.pth \
# --down_ratio 4 --gpus 4   


