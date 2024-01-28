## Original code

https://github.com/xingyizhou/CenterNet

## Requirements

cuda 11.0

torch 1.7

## Getting started

    conda create --name CenterNet python=3.6

    conda activate CenterNet

    ROOT=/path/to/nn-detection

    pip install -r requirements.txt $ROOT

    cd ./source/lib/external

    make

    cd ../datasets/sample

    python setup.py build_ext --inplace

### Training

1. Single-Node multi-process distributed training

::

    >>> python -m torch.distributed.launch --nproc_per_node=4 
               main.py lidar --exp_id your_test_name --batch_size 32 --gpus 0,1,2,3 --num_workers 16  --down_ratio 4  --val_intervals 5 --num_epochs 200  --aug_lidar 0.8

1. Multi-Node multi-process distributed training: (e.g. two nodes)


Node 1: *(IP: 192.168.1.1, and has a free port: 1234)*

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=0 --master_addr="192.168.1.1"
               --master_port=1234 main.py  lidar --exp_id your_test_name --batch_size 32 --gpus 0,1,2,3,4,5,6,7 --num_workers 16  --down_ratio 4  --val_intervals 5 --num_epochs 200  --aug_lidar 0.8

Node 2:

::

    >>> python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE
               --nnodes=2 --node_rank=1 --master_addr="192.168.1.1"
               --master_port=1234 main.py  lidar --exp_id your_test_name --batch_size 32 --gpus 0,1,2,3,4,5,6,7 --num_workers 16  --down_ratio 4  --val_intervals 5 --num_epochs 200  --aug_lidar 0.8

### Infer

    python infer_lidar_od.py lidar --down_ratio 4 --gpus 0 --load_model xx.pth

### Eval

    python saic_od_eva.py --dt_root /home/igs/402/eval/infer_results_0107_model128_5.8w/ --gt_root /home/igs/Desktop/eval_test_data/bin/labels/pull/label/ --Is_class_D 1

    #Is_class_D 0:without class  1:withclass

### Export onnx

    python export_onnx.py lidar --num_stacks 2 --load_model MODEL_PATH.pth

### BBoxes to RVIZ

    sudo apt-get install ros-melodic-jsk-recognition-msgs & sudo apt-get install ros-melodic-jsk-rviz-plugins   ###RVIZ 

    """python lidar128_ros_pipeline.py lidar_od --arch hourglass --down_ratio 4 --gpus 0 --num_stacks 2"""
