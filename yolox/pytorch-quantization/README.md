# Pytorch Quantization

PyTorch-Quantization is a toolkit for training and evaluating PyTorch models with simulated quantization. Quantization can be added to the model automatically, or manually, allowing the model to be tuned for accuracy and performance. Quantization is compatible with NVIDIAs high performance integer kernels which leverage integer Tensor Cores. The quantized model can be exported to ONNX and imported by TensorRT 8.0 and later.

## Install

#### Binaries

```bash
pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

#### From Source

```bash
git clone https://github.com/NVIDIA/TensorRT.git
cd tools/pytorch-quantization
```

Install PyTorch and prerequisites
```bash
https://pytorch.org/get-started/previous-versions/


pip install -r requirements.txt
# for CUDA 10.2 users
pip install torch>=1.8.0
# for CUDA 11.1 users
pip install torch>=1.8.0+cu111    -i https://pypi.tuna.tsinghua.edu.cn/simple 
```

Build and install pytorch-quantization
```bash
# Python version >= 3.7, GCC version >= 5.4 required
python setup.py install

in my env    
python3 setup.py install  --prefix=/home/igs/workspace/pt-dev/lib/python3.6/site-packages
```

#### NGC Container

`pytorch-quantization` is preinstalled in NVIDIA NGC PyTorch container since 20.12, e.g. `nvcr.io/nvidian/pytorch:20.12-py3`

## Resources

* Pytorch Quantization Toolkit [userguide](https://docs.nvidia.com/deeplearning/tensorrt/pytorch-quantization-toolkit/docs/userguide.html)
* Quantization Basics [whitepaper](https://arxiv.org/abs/2004.09602)



#
#
```
(pt-dev) igs@igs:~/workspaces/TensorRT-release-8.0/tools/pytorch-quantization$ pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0


import torch 
torch.cuda.is_available()
```
#
