export YOLOX_DATADIR='./datasets'
export DATA_TYPE='height'
export DEBUG=0   # 1 for True, 0 for False
#
#python tools/train.py -f exps/example/yolox_obb/yolox_obb_s.py -b 4  -c YOLOX_outputs/yolox_obb_s_fp32/epoch_160_ckpt.pth

python tools/train.py -f exps/example/yolox_obb/yolox_obb_s.py -b 4  -c /home/igs/fusion_se_sod_training-QAT/yolox/ptq_out/model-entropy-99.999900.pth