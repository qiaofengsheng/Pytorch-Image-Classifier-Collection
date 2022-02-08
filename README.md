# Pytorch-Image-Classifier-Collection

#### 介绍
==============================

​    支持多模型工程化的图像分类器

==============================

#### 软件架构
Pytorch+opencv

#### 模型支持架构

##### 	模型

|          -           |         -          |         -          |         -          |
| :------------------: | :----------------: | :----------------: | :----------------: |
|       resnet18       |      resnet34      |      resnet50      |     resnet101      |
|      resnet152       |  resnext101_32x8d  |  resnext50_32x4d   |  wide_resnet50_2   |
|   wide_resnet101_2   |    densenet121     |    densenet161     |    densenet169     |
|     densenet201      |       vgg11        |       vgg13        |      vgg13_bn      |
|        vgg19         |      vgg19_bn      |       vgg16        |      vgg16_bn      |
|     inception_v3     |    mobilenet_v2    | mobilenet_v3_small | mobilenet_v3_large |
|  shufflenet_v2_x0_5  | shufflenet_v2_x1_0 | shufflenet_v2_x1_5 | shufflenet_v2_x2_0 |
|       alexnet        |     googlenet      |     mnasnet0_5     |     mnasnet1_0     |
|      mnasnet1_3      |    mnasnet0_75     |   squeezenet1_0    |   squeezenet1_1    |
| efficientnet-b0(0-7) |                    |                    |                    |

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

1. 配置文件config/config.yaml

   ```
   data_dir: "./data/"  #数据集存放地址
   train_rate: 0.8   #数据集划分，训练集比例
   image_size: 128   #输入网络图像大小
   net_type: "shufflenet_v2_x1_0"
   pretrained: True  #是否添加预训练权重
   batch_size: 4   #批次
   init_lr: 0.01   #初始学习率
   optimizer: 'Adam' #优化器 
   class_names: [ 'cat','dog' ]  #你的类别名称，必须和data文件夹下的类别文件名一样
   epochs: 10  #训练总轮次
   loss_type: "mse"  # mse / l1 / smooth_l1 / cross_entropy   #损失函数
   model_dir: "./shufflenet_v2_x1_0/weight/"   #权重存放地址
   log_dir: "./shufflenet_v2_x1_0/logs/"    # tensorboard可视化文件存放地址
   ```

2. 模型训练

   ```
   # 第一次训练
   python train.py
   
   # 接着自己未训练完成的模型继续训练
   python train.py --weights_path 模型保存路径
   ```
   
3. 模型推理

   ```
   # 检测图片
   python infer.py image --image_path 图片地址
   
   # 检测视频
   python infer.py video --video_path 图片地址
   
   # 检测摄像头
   python infer.pu camera --camera_id 摄像头id
   ```
   
4. 部署

   1. onnx打包部署

      ```
      # onnx打包
      python pack_tools/pytorch_to_onnx.py --config_path 配置文件地址 --weights_path 模型权重存放地址
      
      # onnx推理部署
      # 检测图片
      python pack_tools/pytorch_onnx_infer.py image --config_path 配置文件地址 --onnx_path 打包完成的onnx包地址 --image_path 图片地址
      
      # 检测视频
      python pack_tools/pytorch_onnx_infer.py video --config_path 配置文件地址 --onnx_path 打包完成的onnx包地址 --video_path 图片地址
      
      # 检测摄像头
      python pack_tools/pytorch_onnx_infer.py camera --config_path 配置文件地址 --onnx_path 打包完成的onnx包地址 --camera_id 摄像头id，默认为0
      ```

5. 模型剪枝、量化压缩加速

   1. 模型剪枝微调

      ```
      # 模型剪枝微调
      python prune_model/pruning_model.py --weight_path 已训练好的模型权重地址 --prune_type 修剪模型的方式，支持：l1filter,l2filter,fpgm --sparsity 模型稀疏化比例 --finetune_epoches 微调模型的轮次数 --dummy_input 输入模型的形状，例如：(10,3,128,128)
      
      # onnx推理部署
      # 检测图片
      python infer_prune_model.py image --prune_weights_path 剪枝后的模型权重路径  --image_path 图片地址
      
      # 检测视频
      python infer_prune_model.py video --prune_weights_path 剪枝后的模型权重路径  --video_path 图片地址
      
      # 检测摄像头
      python infer_prune_model.py camera --prune_weights_path 剪枝后的模型权重路径  --camera_id 摄像头id，默认为0
      ```


参与贡献

​	作者：qiaofengsheng

​	B站地址：https://space.bilibili.com/241747799

​	github地址：https://github.com/qiaofengsheng/Pytorch-Image-Classifier-Collection.git

​	gitee地址：https://gitee.com/qiaofengsheng/pytorch-image-classifier-collection.git