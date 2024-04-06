
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/6db7b2c9-af2b-44a2-a252-4987804a8108)

在过去两年中， BEV 感知主要出现了四类视图转换模块方案：IPM（Inverse Perspective Mapping）、Lift-splat、MLP（Multi-Layer Perceptron）和Transformer。在这篇回答中，我们将逐一解读。   

## IPM系列    
逆透视变换（Inverse Perspective Mapping，IPM）将透视空间的特征反向映射到BEV空间，实质是`求相机平面与地平面之间的homography矩阵`。IPM假设地面是完美的平面。任何路面高度变化和3D物体颠簸都违反了这一假设。 将图像或者特征映射到地平面会导致强烈的视觉失真，阻碍了在环境中准确定位物体的目标，例如其它车辆和VRU。因此，IPM 转换的图像或者特征通常用于车道检测或自由空间估计，对于这些场景，平面世界假设通常是合理的。
![image](https://github.com/lix19937/dnn-cookbook/assets/38753233/9d3a164d-b00c-4e12-a3da-48b3859c0f77)    
BEV-IPM

Cam2BEV (ICITS 2020)   
Cam2BEV可能不是第一个基于IPM的BEV工作，但是一个高度相关的工作。该方法用IPM进行特征变换，并用CNN来校正不符合路面平坦假设的颠簸。  

## Lift-Splat系列    
Lift-Splat使用单目深度估计，将2D图片特征升维（lift）到成每个相机的视锥体（frustum）特征，并在 BEV 上进行“拍平”（splat）。该方法由Lif-Splat-Shoot（LSS）[4]首次提出，有着 BEV-Seg[5] 、CaDDN [6]、FIERY[7] 、BEVDet[8]、BEVDet4D[9]、M2BEV[10]、BEVFusion[11]和BEVDepth[12]等许多后续工作。
