# HSIC_RPNet
An improved RPNet for HSI classification

## 软件环境
本代码通过MATLAB R2018a软件编写。

使用前，必须先安装一个MATLAB降维算法工具包[drtoolbox](http://lvdmaaten.github.io/drtoolbox/)



## 文件夹介绍
[dataset](https://github.com/YueRP/HSIC_RPNet/tree/master/dataset) 存储所有的数据集文件

[figure](https://github.com/YueRP/HSIC_RPNet/tree/master/figure) 存储程序生成的所有的图片以及论文中的相关实验数据作的折线图(为方便latex作图，以pdf文件保存)

[utils](https://github.com/YueRP/HSIC_RPNet/tree/master/utils) 存储libsvm函数包以及一些子函数实现 

## 代码用途
[Plot_GroundTruth.m](https://github.com/YueRP/HSIC_RPNet/blob/master/Plot_GroundTruth.m) 绘制数据集的地面真值图

[RPNet_Indian_pines_knn.m](https://github.com/YueRP/HSIC_RPNet/blob/master/RPNet_Indian_pines_knn.m) Indian_pines数据集的KNN分类

[RPNet_Indian_pines_svm.m](https://github.com/YueRP/HSIC_RPNet/blob/master/RPNet_Indian_pines_svm.m) Indian_pines数据集的SVM分类

[RPNet_KSC_svm.m](https://github.com/YueRP/HSIC_RPNet/blob/master/RPNet_KSC_svm.m) KSC数据集的SVM分类

[RPNet_Salinas_svm.m](https://github.com/YueRP/HSIC_RPNet/blob/master/RPNet_Salinas_svm.m) Salinas数据集的SVM分类

[RPNet_paviaU_knn.m](https://github.com/YueRP/HSIC_RPNet/blob/master/RPNet_paviaU_knn.m) paviaU数据集的KNN分类

[RPNet_paviaU_svm.m](https://github.com/YueRP/HSIC_RPNet/blob/master/RPNet_paviaU_svm.m) paviaU数据集的SVM分类

[exp_data.xlsx](https://github.com/YueRP/HSIC_RPNet/blob/master/exp_data.xlsx) 记录了所有的实验数据以及相关的原始折线图

## 代码中的参数设置
### [Plot_GroundTruth.m](https://github.com/YueRP/HSIC_RPNet/blob/master/Plot_GroundTruth.m)
此文件可绘制paviaU，Indian Pines，KSC，Salinas四种数据集的地面真值图，可直接运行，不用修改参数。
### 以RPNet开头的matlab脚本文件
在实验时都需要在开头修改参数，具体为：
#### repeat 
实验的重复次数，注意太小则实验结果不具备普遍性，太大则运行时间过长
#### dr
降维方法  可在MDS,PCA,LDA,FA中选择
#### ActivationFunction
激活函数  可选择“LeakyRelu”或“Relu”
#### num_PC 
降维参数 一般取3，可在1-10之间变化
#### Layernum
网络层数 一般取5，可在1-10之间变化

## 程序输出
以RPNet开头的matlab脚本文件会在程序结束后，输出分类器种类，降维方法，降维参数，网络层数，激活函数，重复试验的平均OA，Kappa以及程序运行时间

## [exp_data.xlsx](https://github.com/YueRP/HSIC_RPNet/blob/master/exp_data.xlsx)sheet解释

#### India pines 
India pines数据集的实验数据记录以及折线图，其中num_PC=3，Layernum=5

#### PaviaU 
PaviaU数据集的实验数据记录以及折线图，其中num_PC=3，Layernum=5

#### KSC 
记录了KSC数据集的4组数据

#### Salinas
Salinas数据集的最好效果记录
Salinas数据集训练集与测试集的划分

#### knn
KNN分类器中不同K值的分类数据记录以及折线图（数据集 India_Pines，激活函数 LeakyRelu，降维 MDS）

#### layer_pavia
paviaU数据集不同网络层数下的实验数据及折线图

#### layer_indian
Indian Pines数据集不同网络层数下的实验数据及折线图

#### pcnum_indian
Indian Pines数据集不同降维参数下的实验数据及折线图

#### pcnum_pavia
PaviaU数据集不同降维参数下的实验数据及折线图

####  datanum_accuracy
不同数据集的样本数量对分类效果的相关数据图表
