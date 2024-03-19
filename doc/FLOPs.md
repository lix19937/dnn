
![conv-c](https://github.com/lix19937/dnn-cookbook/assets/38753233/e18e8f26-37ec-4c28-a75c-36c1735edece)
上图表示  
+ 输入shape&emsp;&emsp;&emsp;&emsp; 1 x 3 x 5 x 5    
+ weights shape&emsp;&emsp;&ensp; 2 x 3 x 3 x 3
+ 输出shape&emsp;&emsp;&emsp;&emsp; 1 x 2 x 3 x 3

weights[0, :, :, :]   表示第1组weights   
weights[1, :, :, :]   表示第2组weights   
两组weights 都作用同一个输入    

假设卷积核的尺寸是K×K    
输入：C个特征图  
输出：M个特征图，每个输出的特征图大小为H×W    

则






ref   
https://blog.csdn.net/sinat_28442665/article/details/113738818  
https://blog.csdn.net/liusandian/article/details/79069926   

