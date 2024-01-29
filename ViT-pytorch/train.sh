# python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type ViT-B_16 --pretrained_dir checkpoint/imagenet21k_ViT-B_16.npz   --gradient_accumulation_steps 64


python3 train.py --name cifar10-100_500 --dataset cifar10 --model_type R50-ViT-B_16 --pretrained_dir checkpoint/imagenet21k_R50+ViT-B_16.npz  \
--gradient_accumulation_steps 64  --output_dir output_r50   --num_steps 5