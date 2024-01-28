export YOLOX_DATADIR='./datasets/'
export DATA_TYPE='height'
export DEBUG=0   # 1 for True, 0 for False
#python tools/train.py -f exps/example/yolox_obb/yolox_obb_s.py -d 4 -b 256 #-c /Data/Code/dl/YOLOX_outputs/yolox_obb_s_unfinish/epoch_125_ckpt.pth


#python tools/train.py -f exps/example/yolox_obb/yolox_obb_s.py -d 4 -b 256  --resume -c YOLOX_outputs/calib/model-entropy-99.999900.pth -e 291


#python tools/train.py -f exps/example/yolox_obb/yolox_obb_s.py -d 6 -b 256  -c YOLOX_outputs/calib/model-entropy-99.999900.pth  


python tools/train.py -f exps/example/yolox_obb/yolox_obb_s.py -d 4 -b 128  -c YOLOX_outputs/calib/epoch_499_ckpt.pth  
