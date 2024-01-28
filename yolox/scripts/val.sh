export YOLOX_DATADIR='./datasets'
export DATA_TYPE='height'
export DEBUG=0   # 1 for True, 0 for False
python tools/eval_obb.py -f exps/example/yolox_obb/yolox_obb_s.py -expn eval_obb -d 1 -b 8 --conf 0.05 --nms 0.1 -c /home/igs/epoch_071_ckpt.pth #--trt