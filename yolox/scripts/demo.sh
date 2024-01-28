#export YOLOX_DATADIR='./datasets'
export MODE='demo'
export DATA_TYPE='height'
export DEBUG=1   # 1 for True, 0 for False
# python tools/demo_saic.py -f exps/example/yolox_obb/yolox_obb_s.py --save_result \
# --path ./datasets/val/image_height -c YOLOX_outputs/yolox_obb_s_fp32/epoch_275_ckpt.pth



# python tools/demo_saic.py -f exps/example/yolox_obb/yolox_obb_s.py --save_result \
# --path ./datasets/val/image_height -c  YOLOX_outputs/yolox_obb_s_qat/epoch_300_ckpt.pth



python tools/demo_saic.py -f exps/example/yolox_obb/yolox_obb_s.py --save_result \
--path ./datasets/test/image_height  --trt  --plan  qat1112.onnx._3090.plan