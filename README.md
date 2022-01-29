# Pytorch-Image-Classifier-Collection

#### 介绍
==============================

​    支持多模型工程化的图像分类器

==============================

#### 软件架构
Pytorch+opencv

#### 模型支持架构

##### 	模型

|         -          |         -          |         -          |         -          |
| :----------------: | :----------------: | :----------------: | :----------------: |
|      resnet18      |      resnet34      |      resnet50      |     resnet101      |
|     resnet152      |  resnext101_32x8d  |  resnext50_32x4d   |  wide_resnet50_2   |
|  wide_resnet101_2  |    densenet121     |    densenet161     |    densenet169     |
|    densenet201     |       vgg11        |       vgg13        |      vgg13_bn      |
|       vgg19        |      vgg19_bn      |       vgg16        |      vgg16_bn      |
|    inception_v3    |    mobilenet_v2    | mobilenet_v3_small | mobilenet_v3_large |
| shufflenet_v2_x0_5 | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 |
|      alexnet       |     googlenet      |     mnasnet0_5     |     mnasnet1_0     |
|     mnasnet1_3     |    mnasnet0_75     |   squeezenet1_0    |   squeezenet1_1    |

##### 	损失函数

|  -   |  -   |     -     |       -       |
| :--: | :--: | :-------: | :-----------: |
| mse  |  l1  | smooth_l1 | cross_entropy |

##### 	优化器

|   -    |    -    |    -     |     -      |
| :----: | :-----: | :------: | :--------: |
|  SGD   |  ASGD   |   Adam   |   AdamW    |
| Adamax | Adagrad | Adadelta | SparseAdam |
| LBFGS  |  Rprop  | RMSprop  |            |


#### 安装教程

1.  pytorch>=1.5即可，其余库自行安装即可。

#### 使用说明

1. 模型训练

   ```
   第一次训练
   python train.py
   接着自己未训练完成的模型继续训练
   python train.py --weights_path 模型保存路径
   ```

2. 模型推理

   ```
   python infer.py
   ```

#### 参与贡献

​	作者：qiaofengsheng

​	B站地址：https://space.bilibili.com/241747799

​	github地址：https://github.com/qiaofengsheng/Pytorch-Image-Classifier-Collection.git

​	gitee地址：https://gitee.com/qiaofengsheng/pytorch-image-classifier-collection.git
