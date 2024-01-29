**libspconv** A tiny `inference engine`, only supports sm_80 & sm_86 on Tesla Platform and sm_87 on Embedded Platform.

libprotobuf-dev == 3.6.1
```
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git checkout v3.6.1
git submodule update --init --recursive
./autogen.sh
./configure
make -j$(nproc)
make -j$(nproc) check
sudo make install
```
https://github.com/NVIDIA-AI-IOT/Lidar_AI_Solution/tree/master/libraries/3DSparseConvolution#note   

* The current version supports compute arch are required sm_80, sm_86, and sm_87..    
* Supported operators:     
SparseConvolution, Add, Relu, Add&Relu, ScatterDense, Reshape and ScatterDense&Transpose.  

* Supported SparseConvolution:    
SpatiallySparseConvolution and SubmanifoldSparseConvolution.

* Supported properties of SparseConvolution:    
activation, kernel_size, dilation, stride, padding, rulebook, subm, output_bound, precision and output_precision.

支持有限shape  
```


构建一个SparseConvolution 组成的onnx 进行推理验证，不依赖trt    
