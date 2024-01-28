  export CUDA_VISIBLE_DEVICES="4,5,6,7"

  python export_onnx.py lidar_od_seg --dataset segmentation --load_model ../exp/lidarv2/default/model_best.pth

