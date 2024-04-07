
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/6db7b2c9-af2b-44a2-a252-4987804a8108)

![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/4e0b1daf-f721-4503-8732-580d59c43baf)

![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/152317b7-e55b-4562-811c-cb09a8039f27)



在过去两年中， BEV 感知主要出现了四类视图转换模块方案：IPM（Inverse Perspective Mapping）、**Lift-splat**、MLP（Multi-Layer Perceptron）和 **Transformer**。      

## IPM系列    
逆透视变换（Inverse Perspective Mapping，IPM）将透视空间的特征反向映射到BEV空间，实质是`求相机平面与地平面之间的homography矩阵`。IPM假设地面是完美的平面。任何路面高度变化和3D物体颠簸都违反了这一假设。 将图像或者特征映射到地平面会导致强烈的视觉失真，阻碍了在环境中准确定位物体的目标，例如其它车辆和VRU。因此，IPM 转换的图像或者特征通常用于车道检测或自由空间估计，对于这些场景，平面世界假设通常是合理的。
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/9d3a164d-b00c-4e12-a3da-48b3859c0f77)    
BEV-IPM[1]    

### Cam2BEV (ICITS 2020)[2]          
Cam2BEV可能不是第一个基于IPM的BEV工作，但是一个高度相关的工作。该方法用IPM进行特征变换，并用CNN来校正不符合路面平坦假设的颠簸。  

### VectorMapNet (2022/06, Arxiv)[3]    
VectorMapNet也同样指出在不知道地平面的确切高度的情况下，仅用homography矩阵进行变换是不准确的。为了缓解这个问题，作者进一步将图像特征转换为四个具有不同高度（-1m、0m、1m、2m）的BEV平面。   

## Lift-Splat系列    
Lift-Splat使用单目深度估计，将2D图片特征升维（lift）到成每个相机的视锥体（frustum）特征，并在 BEV 上进行“拍平”（splat）。该方法由Lif-Splat-Shoot（LSS）[4]首次提出，有着 BEV-Seg[5] 、CaDDN [6]、FIERY[7] 、BEVDet[8]、BEVDet4D[9]、M2BEV[10]、BEVFusion[11]和BEVDepth[12]等许多后续工作。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/43ccc160-ca62-49d3-8c9c-bad3ab450ab9)  

### M2BEV (2022/04, Arxiv)[10]    
Lift-Splat-Shoot为估计每个视锥体voxel的深度分布，耗费了大量显存，继而限制了backbone的大小。为了节省显存使用，M2BEV[9]假设沿射线的深度分布是均匀的，也就是沿相机射线的所有voxel都填充有与 2D 空间中的单个像素对应的相同特征。这个假设通过减少学习参数的数量来提高计算和内存效率。GPU显存占用仅为原始版本的 1/3，因此可以使用更大的backbone以获得更好的性能。   

### BEVFusion (2022/05, Arxiv)[11]    
为了实现splat操作，Lift-Splat-Shoot[4]利用“Cumulative Sum(CumSum) Trick”，根据其对应的 BEV 网格 ID 对所有视锥体特征进行排序，对所有特征执行累积求和，然后减去边界处的累积求和值。 然而，“CumSum Trick”存在两个缺陷损害探测器的整体运行速度：    
涉及对大量 BEV 坐标的排序过程，增加额外的计算量；       
采用的Prefix Sum技术使用串行方式计算，因此运行效率低下。   

![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/a701418c-a57e-46d1-bc9b-a9fc64655f3f)

因此，BEVFusion[11]优化了BEV pooling中网格关联和特征聚合，将其加速了40倍。其中：    
网格关联的目标是将每个视锥体特征的 3D 坐标和 BEV 网格建立索引，可以通过缓存预先计算和排序的结果，降低网格关联延迟。    
特征聚合的目标是通过对称函数（例如mean、max和sum等）聚合每个 BEV 网格内的特征。为了并行化，每个BEV网格可以分配一个GPU线程，并设计专用 GPU Kernel加速。    

### BEVDepth (2022/06, Arxiv)[12]     
类似BEVFusion的并行化思路，BEVDepth[12]则为每个视锥体特征分配一个GPU线程，并行化版本替换原来的BEV pooling模块可以加快80倍，算法整体也加速了3倍。因为每个视锥体特征是等长的，所以并行程度更高。    
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/0dd07f39-767c-419d-a394-c457478ac289)   
BEVDepth[12]   

## MLP系列     
通过MLP对视图转换进行建模，也能学习透视空间到BEV空间的映射关系。这类方案由 VPN[13]发起，Fishing Net [14]、PON[15]和 HDMapNet[16]紧随其后。   

### VPN (RAL 2020) [13]    
VPN将 BEV的2D物理范围拉伸为1维向量，然后对其执行全连接操作。换句话说，它忽略了强几何先验，而纯粹采用数据驱动的方法来学习Perspective View到BEV的变换。这种变换是特定于相机的，因此每个相机都必须学习一个网络，往往参数较多，有一定的过拟合风险。

### PON (CVPR 2020 oral) [15]     
考虑到几何先验，PON先收缩图像特征的垂直维度（通道维度映射到大小为B），但保留水平维度 W；然后沿水平轴并reshape特征图成维度为C×Z×W的张量，最后基于已知的相机焦距重采样成笛卡尔坐标系的BEV特征。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/d721f989-de7f-46fb-af60-26a22de21976)   
PON[15]   

#### MLP系列的优缺点    
MLP系列的优点很明显，实现非常简单，也很容易在车端部署。但是缺点也很明显，相机的内外参是重要的先验信息（inductive bias），MLP放弃掉这些有用的信息，而采取数据驱动的方式隐式学习内外参，将其融入到MLP的权重当中，有点舍近求远，性能上和后续的Transformer系列相比也有更低的天花板。     

## Transformer系列      
自 2020 年年中以来，transformer[17] 席卷计算机视觉领域，使用基于attention的transformer对视图转换进行建模显示出吸引力。由于使用全局注意力机制，transformer 更适合执行视图转换的工作。目标域中的每个位置具有相同的距离来访问源域中的任何位置，克服了 CNN 中卷积层感受野受限局部。     

Transformer 中有两种注意力机制，encoder 中的 self attention 和 decoder 中的 cross attention。它们之间的主要区别是query Q。在 self attention 中，Q、K、V 输入是相同的，而在 cross attention 中，Q 与 K 和 V 的域不同，attention 模块的输出尺寸与query Q 相同。简而言之，self attention 可以看作是原始特征域中的特征增强器，而 cross attention 则可以被视为跨域生成器。  
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/88c835cb-b322-4973-bcc8-ba228a17b52b)

Transformers 的许多最新进展实际上仅利用了self attention机制，例如被大量引用的 ViT或 [18]Swin Transformer[19]。它们用于增强backbone提取的特征。然而，考虑到在量产车上嵌入式系统资源有限，部署 Transformer 存在困难。相对于容易部署的CNN，self attention的增量收益较小。因此，在self attention机制取得突破性优势之前，量产自动驾驶使用CNN会是一个明智的选择。    

### DETR[20]     
另一方面，使用cross attention理由更为充分和可靠。将cross attention应用于计算机视觉的一项开创性研究是 DETR[20]。 DETR最具创新性的部分之一是object query，即基于固定数量槽的cross-attention decoder。原始的 Transformer 论文将每个query逐个自回归输入decoder，但DETR将这些query并行输入到 DETR decoder中。除了query的数量，query的内容是学习的，不需要在训练前指定。Query可以被视为预先分配的模板来保存对象检测结果，cross-attention decoder完成填充空白的工作。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/cf9278d8-7b9f-4c6f-91ea-88967b316bee)  
DETR  

这提示了使用cross-attention decoder进行视图转换的想法。输入视图被送入特征编码器（基于self-attention或基于CNN），编码后的特征作为K和V。目标视图格式的query Q可以学习，只需要栅格化为模板。Q 的值可以与网络的其余部分一起学习。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/16923f72-cf4c-4fcd-b905-4dbf600b963a)    

回顾一些最相关的工作，并探讨在特斯拉AI Day上Andrej Karpathy 分享的特斯拉 FSD 中transformer的使用[21]。    

### PYVA (CVPR 2021)[22]      
PYVA[22]是第一个明确提到cross-attention decoder可用于视图转换以将图像特征提升到 BEV 空间的方法之一。PYVA 首先使用 MLP 将透视空间中的图像特征 X 提升到BEV 空间中的 X'。第二个 MLP 将 X' 映射回图像空间 X''，并使用 X 和 X' 之间的循环一致性损失来确保此映射过程保留尽可能多的相关信息。PYVA使用的Transformer是一个cross-attention模块，query Q要映射到BEV空间中的BEV特征X'。  
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/6573553f-f804-45c6-b984-a9e5c622358b)   
PYVA   

由于没有对 BEV 空间中生成的query 的明确监督，从技术上讲，很难将 MLP 和cross attention这两个组件的贡献区分开来。对此，进一步的消融研究将有助于澄清这一点。

### NEAT（ICCV 2021）[23]      
NEAT[23]使用 Transformer 增强图像特征空间中的特征，然后使用基于 MLP 的迭代注意力将图像特征提升到 BEV 空间中。Encoder 块中使用的 Transformer 是基于self attention的。最有趣的部分发生在神经注意力领域 (NEAT) 模块中。对于给定的输出位置 (x, y)，使用 MLP 将输出位置和图像特征作为输入，生成与输入特征图像空间维度相同的注意力图。然后使用注意力图对原始图像特征进行点积，以生成给定输出位置的目标 BEV 特征。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/37b78744-9b9a-4093-92b9-62b09d0da582)     
NEAT     
NEAT 模块与cross-attention机制相似，主要区别在于 Q 和 K 之间的相似性测量步骤由 MLP 代替。    
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/98f8839d-d763-461b-9678-277378175eb9)

### STSU (ICCV 2021)[24]      
STSU[24]遵循 DETR 的做法，使用稀疏查询进行对象检测。 STSU 不仅可以检测动态对象，还可以检测静态道路布局。  
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/323e9247-06e3-4ca6-b78a-eaa4ef03c346)    
STSU   

### DETR3D (CoRL 2021)[25]    
DETR3D[25]也使用稀疏查询进行对象检测，与 STSU 类似，但 DETR3D 侧重于动态物体。Query位于 BEV 空间中，它们使 DETR3D 能够直接在 BEV 空间中预测，而不是对图像特征进行密集变换。    
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/9006b9ed-28c4-4eca-b6bc-4f0c68f9b069)   
DETR3D   

### CVT (CVPR 2022 oral) [26]
CVT[26]同样使用cross-view cross-attention机制将多尺度特征聚合成统一的BEV表示。 cross-view cross-attention依赖于位置嵌入，是各个相机图像的2D位置在深度等于1情况下，利用相机内外参数反投影到统一的3D坐标，然后通过MLP生成。它包含场景的几何结构，并学习匹配透视视图和BEV位置。最后将位置嵌入与图像特征结合在cross-view cross-attention的key中，使得能同时使用外观和几何线索来推理不同视图之间的对应关系。    
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/18027f62-850b-4496-b570-edfd1a238641)   
CVT   

### PETR (ECCV 2022)[27]      
DETR3D 为端到端 3D 对象检测提供了直观的解决方案，但依然存在2个问题：

参考点的预测坐标可能不准确，使采样的特征超出目标区域；
投影点处的图像特征，无法从全局视图中学习，复杂的特征采样会阻碍算法的实际应用。
PETR[27]认为在特征转换过程中使用显式 2D-3D 投影会阻碍网络执行全局推理的能力（注：笔者不一定同意）。相反，它使用 3D 位置嵌入（3D Positional Embedding, 3D PE）来促进全局推理，并通过为 2D 图像提供 3D 位置嵌入来要求神经网络隐式学习采样位置。通过这种嵌入，对应于相同 3D 区域的 2D 区域将具有相似的 3D 嵌入。不同于CVT中的伪3D PE（深度等于1），PETR中3D PE是对3D感知范围的稠密采样。
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/734d9715-e3ea-484f-a459-19b7a5f06a6d)   
PETR    

### PETRv2 (2022/06, Arxiv)[28]      
不同于PETR 中的 3D PE 独立于输入图像，PETRv2 [27]指出3D PE应该由2D图像特征驱动，因为2D图像特征可以提供指导（例如，深度信息）。PETRv2提出了一个特征引导的位置编码器，它隐含地引入了视觉先验，2D 图像特征被输入到一个小型MLP网络中，用于生成3D PE。  
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/a0b391a3-aeea-4c8f-b7ab-cad774a7e06a)    
PETRv2     
PETRv2因为分割任务，才真正引入了BEV空间的query，每个分割query对应于一个特定BEV patch。分割query在BEV空间中使用固定锚点初始化，类似于 PETR中检测query的生成。然后通过MLP将这些锚点投影到分割query中。之后，分割query被输入到transformer解码器并与图像特征交互。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/e52d50f8-24b0-47fa-b82b-aefc639dea1f)   
PETRv2      

### Translating Images into Maps（ICRA 2022 best paper）[29]     
Translating Images into Maps [29]发现无论图像像素的深度如何，图像中的垂直扫描线（图像列）与通过 BEV 地图中相机位置的极射线之间存在 1-1 对应关系。这类似于 OFT (BMVC 2019) 和 PON (CVPR 2020) 的想法，它们沿着投射回 3D 空间的光线在像素位置涂抹特征。在列方向使用轴向cross-attention transformer 和在行方向使用卷积可以显着节省计算量。   
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/2626bc0f-6e5f-41cc-bb3d-2360ee4abd1a)   

### 特斯拉方案（AI Day 2021）[21]     
在 2021 年的特斯拉 AI Day，特斯拉揭示了为 FSD采用的神经网络的丰富细节 [21]。最有趣的构建模块之一是被称为“图像到 BEV 转换 + 多相机融合”的模块。中心是一个 transformer 模块，或者更具体地说，一个cross attention模块。   

![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/f4dd4b3b-d3b6-4187-b6aa-f02b9d7ad657)   

```
初始化一个你想要的输出空间大小的栅格，然后在输出空间中用正弦和余弦的位置编码平铺它，然后用 MLP 将它们编码成一组query向量，然后全部图像及其特征也发出自己的key和value，然后query key和value输入multi-headed self-attention模块（笔者注：这实际上是交叉注意力）。

— Andrej Karpathy [21]     
```

虽然 Andrej 提到他们使用了 multi-headed self attention，但他所描述的显然是一种cross attention机制，而且他幻灯片中的右边图表也指向了原始transformer论文中的cross attention模块。

此视图转换中最有趣的部分是 BEV 空间中的查询。它由 BEV 空间中的栅格生成（如 DETR，空白且预分配模板），并与位置编码 (PE) 连接。还有一个上下文摘要，它使用位置编码平铺。该图没有显示上下文摘要如何生成以及如何与位置编码一起使用的详细信息，但笔者认为可能存在一个global pooling可以折叠透视空间中的所有空间信息，以及一个平铺操作将这个 1x1 张量平铺在预定义的 BEV 网格。   
笔者列出了视图转换模块中更详细的模块（圆圈）和对应的张量及其形状（正方形）。BEV 空间中的张量用蓝色标记，核心cross attention模块用红色标记。  
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/a18be445-53d8-4272-ab67-13a27c9693a8)  

### BEVFormer (ECCV 2022) [30]     
BEVFormer 通过预定义的网格形 BEV query、spatial cross attention（SCA）和temporal self attention（实际上也是cross attention）交互时空信息。SCA对于 BEV 网格中的每个pillar，沿高度从-5m 到3m，每2m采样一次共4个点，并投影到图像以形成参考点。   ![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/a3f537fe-2543-4bd7-a89b-259c74b26293)   
BEVFormer     
BEVFormer使用了Deformable DETR 中提出的 Deformable attention。DETR 的问题是长时间训练收敛慢和检测小物体的性能低下。因此，Deformable DETR 首先通过只关注参考周围的一小组关键采样点来减少计算量。然后它使用多尺度可变形注意力模块来聚合多尺度特征（没有 FPN）来帮助小目标检测。每个object query仅限于关注参考点周围的一小组关键采样点，而不是特征图中的所有点。

### PersFormer (ECCV 2022 oral) [31]      
PersFormer采用统一的 2D/3D anchor设计和辅助任务同时检测 2D/3D 车道线。视角变换的总体思路是先使用来自 IPM 的坐标变换矩阵作为参考，通过关注前视图特征中的相关区域（局部上下文）来生成 BEV 特征表示。这与BEVFormer 非常相似，区别PersFormer是通过IPM将参考点固定在地面上（设置 z=0）。     

值得一提的是，PersFormer在Waymo Open数据集 [32]的基础上，提出了3D车道线新数据集OpenLane。数据集使用 2D 和 3D 激光雷达点云进行注释。 每个图像中都标记了车道的可见性，并且似乎比带有高清地图的 AutoLabel 方法（nuScenes 数据集 [33]）要精细得多。    

## 总结      
以上4类方法各有优势，同时也存在一些挑战：   
+ IPM的平坦地面假设过于严格，因此通常只用于车道检测或自由空间估计。    
+ `Lift-Splat`使用额外的网络估计深度，耗费大量显存，且限制其它模块大小，影响整体性能。    
+ MLP 的收益平衡点需要考虑数据量、GPU资源和工程工作。   
+ `Transformers` 的数据依赖性使其更具表现力，但也难以训练。另外，在量产自动驾驶汽车资源有限的嵌入式系统中部署 Transformer 也可能是一个重大挑战。    

面对这些挑战，尽管可能存在遗漏，笔者最大程度上尝试梳理了现有BEV视角变换的研发脉络，反馈给整个量产自动驾驶community。希望有助于感兴趣的读者在这个方向上更深入地挖掘。

关注我们，之后还会进一步分享关于BEV下的multi-task联合训练，时序信息的利用，以及如何做联合感知和预测。也欢迎各位加入小鹏一起持续提升这个富有挑战性的新范式！

## 参考      
^Kim, Youngseok, and Dongsuk Kum. "Deep learning based vehicle position and orientation estimation via inverse perspective mapping image." 2019 IEEE Intelligent Vehicles Symposium (IV). IEEE, 2019. https://ieeexplore.ieee.org/document/8814050
^Reiher, Lennart, Bastian Lampe, and Lutz Eckstein. "A sim2real deep learning approach for the transformation of images from multiple vehicle-mounted cameras to a semantically segmented image in bird’s eye view." 2020 IEEE 23rd International Conference on Intelligent Transportation Systems (ITSC). IEEE, 2020. https://arxiv.org/pdf/2005.04078
^Liu, Yicheng, et al. "VectorMapNet: End-to-end Vectorized HD Map Learning." arXiv preprint arXiv:2206.08920 (2022). https://arxiv.org/pdf/2206.08920
^abPhilion, Jonah, and Sanja Fidler. "Lift, splat, shoot: Encoding images from arbitrary camera rigs by implicitly unprojecting to 3d." European Conference on Computer Vision. Springer, Cham, 2020. https://arxiv.org/pdf/2008.05711
^Ng, Mong H., et al. "BEV-Seg: Bird's Eye View Semantic Segmentation Using Geometry and Semantic Point Cloud." arXiv preprint arXiv:2006.11436 (2020). https://arxiv.org/pdf/2006.11436 [6] Reading, Cody, et al. "Categorical depth distribution network for monocular 3d object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021. https://arxiv.org/abs/2103.01100
^Reading, Cody, et al. "Categorical depth distribution network for monocular 3d object detection." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021. https://arxiv.org/abs/2103.01100
^Hu, Anthony, et al. "FIERY: Future Instance Prediction in Bird's-Eye View From Surround Monocular Cameras." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. https://arxiv.org/pdf/2206.04584
^Huang, Junjie, et al. "Bevdet: High-performance multi-camera 3d object detection in bird-eye-view." arXiv preprint arXiv:2112.11790 (2021). https://arxiv.org/pdf/2112.11790
^abHuang, Junjie, and Guan Huang. "Bevdet4d: Exploit temporal cues in multi-camera 3d object detection." arXiv preprint arXiv:2203.17054 (2022). https://arxiv.org/pdf/2203.17054
^abXie, Enze, et al. "M^2BEV: Multi-Camera Joint 3D Detection and Segmentation with Unified Birds-Eye View Representation." arXiv preprint arXiv:2204.05088 (2022). https://arxiv.org/abs/2204.05088
^abcLiu, Zhijian, et al. "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation." arXiv preprint arXiv:2205.13542 (2022). https://arxiv.org/pdf/2205.13542
^abcdLi, Yinhao, et al. "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection." arXiv preprint arXiv:2206.10092 (2022). https://arxiv.org/abs/2206.10092
^abPan, Bowen, et al. "Cross-view semantic segmentation for sensing surroundings." IEEE Robotics and Automation Letters 5.3 (2020): 4867-4873. https://arxiv.org/pdf/1906.03560
^Hendy, Noureldin, et al. "Fishing net: Future inference of semantic heatmaps in grids." arXiv preprint arXiv:2006.09917 (2020). https://arxiv.org/pdf/2006.09917
^abcRoddick, Thomas, and Roberto Cipolla. "Predicting semantic map representations from images using pyramid occupancy networks." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020. https://arxiv.org/abs/2003.13402
^Li, Qi, et al. "Hdmapnet: A local semantic map learning and evaluation framework." arXiv preprint arXiv:2107.06307 (2021). https://arxiv.org/pdf/2107.06307
^Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems 30 (2017). https://arxiv.org/abs/1706.03762
^Dosovitskiy, Alexey, et al. "An image is worth 16x16 words: Transformers for image recognition at scale." arXiv preprint arXiv:2010.11929 (2020). https://arxiv.org/abs/2010.11929
^Liu, Ze, et al. "Swin transformer: Hierarchical vision transformer using shifted windows." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. https://arxiv.org/abs/2103.14030
^Carion, Nicolas, et al. "End-to-end object detection with transformers." European conference on computer vision. Springer, Cham, 2020. https://arxiv.org/pdf/2005.12872.pdf
^abcdKarpathy, Andrej, et al. "Tesla AI Day" YouTube, uploaded by Tesla, 20 Aug. 2021. https://youtu.be/j0z4FweCy4M?t=3613
^abYang, Weixiang, et al. "Projecting your view attentively: Monocular road scene layout estimation via cross-view transformation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2021. https://ieeexplore.ieee.org/document/9578824
^abChitta, Kashyap, Aditya Prakash, and Andreas Geiger. "Neat: Neural attention fields for end-to-end autonomous driving." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. https://arxiv.org/abs/2109.04456
^abCan, Yigit Baran, et al. "Structured Bird's-Eye-View Traffic Scene Understanding From Onboard Images." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. https://arxiv.org/abs/2110.01997
^abWang, Yue, et al. "Detr3d: 3d object detection from multi-view images via 3d-to-2d queries." Conference on Robot Learning. PMLR, 2022. https://arxiv.org/abs/2110.06922
^abZhou, Brady, and Philipp Krähenbühl. "Cross-view Transformers for real-time Map-view Semantic Segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022. https://arxiv.org/abs/2205.02833
^abcLiu, Yingfei, et al. "Petr: Position embedding transformation for multi-view 3d object detection." European conference on computer vision. 2022. https://arxiv.org/abs/2203.05625
^Liu, Yingfei, et al. "PETRv2: A Unified Framework for 3D Perception from Multi-Camera Images." arXiv preprint arXiv:2206.01256 (2022). https://arxiv.org/abs/2206.01256
^abSaha, Avishkar, et al. "Translating images into maps." International Conference on Robotics and Automation. 2022
^Li, Zhiqi, et al. "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers." European conference on computer vision. 2022. https://arxiv.org/pdf/2203.17270
^Chen, Li, et al. "PersFormer: 3D Lane Detection via Perspective Transformer and the OpenLane Benchmark." European conference on computer vision. 2022.  https://arxiv.org/pdf/2203.11089
^Ettinger, Scott, et al. "Large scale interactive motion forecasting for autonomous driving: The waymo open motion dataset." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021. https://arxiv.org/abs/2104.10133
^Caesar, Holger, et al. "nuscenes: A multimodal dataset for autonomous driving." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020. https://arxiv.org/abs/1903.11027

## ref    
https://www.zhihu.com/question/521842610












