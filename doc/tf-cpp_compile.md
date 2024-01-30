## linux下    
+ 安装bazel     
源码安装  可能依赖其他库 比如protoc    
```
unzip ./bazel.0.11.1-dist.zip  -d ./bazel.0.11.1-dist
cd ./bazel.0.11.1-dist
./compile.sh
cp output/bazel  /usr/local/bin
```

二进制安装  推荐    
sudo sh ./bazel-0.11.1-installer-linux-x86_64.sh      验证 bazel --help

拓展   
If you get the error : Closure Rules requires Bazel >=0.4.5 but was 0.11.1 , see like this bazel#4834. Change all the error file the version 0.4.5 to 0.0.0. Then you will build it successfully.
_check_bazel_version("Closure Rules", "0.4.5")

另参考https://www.jianshu.com/p/d53d231e8cf0    
or   
二进制文件bazel-0.11.0-installer-linux-x86_64.sh (包含了jdk)   
sh ./bazel-0.11.0-installer-linux-x86_64.sh    


安装python 2.7/3.5  一般centos系统会默认安装好了2.7  重要勿删    
安装numpy  six  wheel  python-dev /python-devel(centos)    
```
pip install  numpy  six  wheel  
yum install python-devel   zib-devel
```
如果pip没有预装    
```
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py   # 下载安装脚本
python get-pip.py         # 运行安装脚本
or  python3 get-pip.py    # 运行安装脚本
pip --version
```   
https://www.runoob.com/w3cnote/python-pip-install-usage.html
/usr/lib/python2.7/site-packages
pip freeze命令可以查看用pip安装的软件有哪些
要查看安装路径,在执行一次命令pip install xx，就会告诉你已经安装，安装路径在哪

yum install python-devel   
查看位置  
```
rpm -qa| grep  python-devel
rpm -ql python-devel-2.7.5-86.el7.x86_64
```

configure配置  
```
cd  tensorflow
./configure 
```
有cpu/gpu 相关选项选择
GPU在build Tensorflow with CUDA support ? 选项中一定要选择Y，并输入对应的CUDA以及cuDNN版本

gpu算力 https://developer.nvidia.com/cuda-gpus#compute
https://www.nvidia.cn/Download/Find.aspx?lang=cn


## 进入根目录后编译 # 编译生成.so文件, 编译C++ API的库 (建议) 
cpu   
bazel build --config=opt  //tensorflow:libtensorflow_cc.so --verbose_failures
 
#也可以选择,编译C API的库    
bazel build --config=opt  //tensorflow:libtensorflow.so

or gpu    
bazel build --config=opt  --config=cuda  //tensorflow:libtensorflow_cc.so  --verbose_failures

生成gen目录    
sh  tensorflow/contrib/makefile/build_all_linux.sh

+ 下载依赖   
注意：看下路径     ./tensorflow/tensorflow/contrib/makefile下有没有downloads文件夹。如果没有的话需要在./tensorflow/tensorflow/contrib/makefile文件夹下打开终端执行一个sh脚本文件
生成downloads目录
sh  ./tensorflow/contrib/makefile/download_dependencies.sh   会下载protoc

若出现如下错误 /autogen.sh: 4: autoreconf: not found ，安装相应依赖即可 sudo apt-get install autoconf automake libtool
```
mkdir  /usr/local/tf-gpu/include 
sudo cp -r bazel-genfiles/  /usr/local/tf-gpu/include 
sudo cp -r tensorflow  /usr/local/tf-gpu/include 
sudo cp -r third_party  /usr/local/tf-gpu/include 
sudo cp -r bazel-bin/tensorflow/libtensorflow_cc.so  /usr/local/tf-gpu/lib/
sudo cp -r bazel-bin/tensorflow/ libtensorflow_framework.so.so  /usr/local/tf-gpu/lib/
```

https://tensorflow.google.cn/install/source      

Tensorflow版本和cuda/cudnn版本的对应关系
https://blog.csdn.net/zhangyonghui007/article/details/93603024

openjdk环境变量设置
```
　　export JAVA_HOME=/home/lix/tools/jdk1.8.0_171 
　　export JRE_HOME=$JAVA_HOME/jre 
　　export CLASSPATH=.:$JAVA_HOME/lib:$JRE_HOME/lib
　　export PATH=$JAVA_HOME/bin:$PATH   
```

http://us.download.nvidia.com/XFree86/Linux-x86_64/396.37/NVIDIA-Linux-x86_64-396.37.run
http://cn.download.nvidia.com/tesla/396.37/NVIDIA-Linux-x86_64-396.37.run
https://us.download.nvidia.cn/XFree86/Linux-x86_64/396.37/NVIDIA-Linux-x86_64-396.37.run

-silent –driver

undefined symbol: _ZN10tensorflow8internal21CheckOpMessageBuilder9NewStringEv
https://blog.csdn.net/duinodu/article/details/71788484



# windows下   

github branch下载     
设置系统环境变量  PreferredToolArchitecture=x64     验证
安装python3.6  pip numpy  six  验证    
安装swigwin-4.0.1 最好是二进制安装    
cmake  配置      

https://zhuanlan.zhihu.com/p/30528874
 




版本对应关系  及其源码编译安装
https://www.tensorflow.org/install/source#common_installation_problems

包安装
https://www.tensorflow.org/install/gpu
比如
pip3 install tensorflow-gpu==1.8

安装whl
pip install  ./tensorflow_gpu-1.8.0-cp36-cp36m-manylinux1_x86_64.whl

安装opencv-python












windows 编译cpython
linux  编译cypthon




cuda默认安装路径  C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\
vm.n
查看版本
nvcc --version
include/cudnn.h

https://developer.nvidia.com/rdp/cudnn-archive
642061776@qq.com
lix520wy+

编译win-tensorflow C++
dos下 输入：set PreferredToolArchitecture=x64
验证echo %PreferredToolArchitecture%


参考https://blog.csdn.net/h8832077/article/details/78988488
0,cmake-gui配置
cmake  C:/Users/T5810/Downloads/tensorflow-r1.5/tensorflow/contrib/cmake    E:/winlix/tf_v1.5_build
注意 vs安装时把git相关的组件加入

1,build 中出现timeout  grpc （ Failed to connect to boringssl.googlesource.com port 443: Timed out）
grpc 和hdfs需要翻墙下载

vs出现git clone https://boringssl.googlesource.com/boringssl
https://blog.csdn.net/qq_33487412/article/details/78458000
配置git http port  port具体见lan端口   git config --global http.proxy 127.0.0.1:1080

2,找不到device_attributes.pb_text.h
是后来下载生成的


crnn预测一张词条 [GTX1080]   
           用时/ 词条数目   AVG 
tf-cpu 版本 245s / 1748      0.14  
tf-gpu 版本 107s / 1748      0.06
gpu / cpu  = 3/7

https://blog.csdn.net/hjimce/article/details/47323463










gcc --std=c++11 -DIS_SLIM_BUILD -fno-exceptions -DNDEBUG -O3 -march=native -fPIC -I. -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/downloads/ -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/downloads/eigen -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/downloads/gemmlowp -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/downloads/nsync/public -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/downloads/fft2d -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/proto/ -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/proto_text/ -I/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/protobuf-host/include -I/usr/local/include \
-o /home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/bin/benchmark /home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/obj/tensorflow/core/util/reporter.o /home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/obj/tensorflow/tools/benchmark/benchmark_model.o /home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/obj/tensorflow/tools/benchmark/benchmark_model_main.o \
 -L/home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/protobuf-host/lib -Wl,--allow-multiple-definition -Wl,--whole-archive  /home/poc/tensorflow-r1.5/tensorflow/contrib/makefile/gen/lib/libtensorflow-core.a -Wl,--no-whole-archive tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11/nsync.a -lstdc++ -lprotobuf -lz -lm -ldl -lpthread
 

https://blog.csdn.net/Mundane_World/article/details/81636609?utm_medium=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase&depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromMachineLearnPai2-1.nonecase

https://www.cnblogs.com/chaofn/p/9309560.html


 
 
 https://github.com/brohrer
 http://brohrer.github.io/
 http://brohrer.github.io/blog.html
 https://brohrer.mcknote.com/zh-Hans/how_machine_learning_works/how_linear_regression_works.html
 
 
【内存占用】
https://blog.csdn.net/lzrtutu/article/details/81079861 https://github.com/tensorflow/tensorflow/blob/16fabd1fc3408252d816bf53698df8f05c9ee304/tensorflow/core/common_runtime/gpu/gpu_device_test.cc#L30-L49
 per_process_gpu_memory_fraction 
 TEST(GPUDeviceTest, VirtualDeviceConfigConflictsWithMemoryFractionSettings) {
  SessionOptions opts = MakeSessionOptions("0", 0.1, 1, {{}});
  std::vector<tensorflow::Device*> devices;
  Status status = DeviceFactory::GetFactory("GPU")->CreateDevices(
      opts, kDeviceNamePrefix, &devices);
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_TRUE(StartsWith(status.error_message(),
                         "It's invalid to set per_process_gpu_memory_fraction"))
      << status;
}



https://github.com/yongyehuang 
https://www.captainbed.net/ 
https://github.com/chiphuyen/stanford-tensorflow-tutorials
