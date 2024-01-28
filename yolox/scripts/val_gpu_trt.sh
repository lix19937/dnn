export YOLOX_DATADIR='./datasets/'
export DATA_TYPE='height'
export DEBUG=0   # 1 for True, 0 for False
python tools/eval_obb_trt.py --trt --plan_file qat1112.onnx._3090.plan \
-f exps/example/yolox_obb/yolox_obb_s.py \
 -expn eval_obb -d 1 -b 1 --conf 0.05 --nms 0.1  #--trt
