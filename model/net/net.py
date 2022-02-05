'''
 _*_coding:utf-8 _*_
 @Time     :2022/1/28   19:05
 @Author   : qiaofengsheng
 @File     :net.py
 @Software :PyCharm
 '''

import torch
from torchvision import models
from torch import nn
from efficientnet_pytorch import EfficientNet

class ClassifierNet(nn.Module):
    def __init__(self, net_type='resnet18', num_classes=10, pretrained=False):
        super(ClassifierNet, self).__init__()
        self.layer = None
        if net_type == 'resnet18': self.layer = nn.Sequential(models.resnet18(pretrained=pretrained), )
        if net_type == 'resnet34': self.layer = nn.Sequential(models.resnet34(pretrained=pretrained), )
        if net_type == 'resnet50': self.layer = nn.Sequential(models.resnet50(pretrained=pretrained), )
        if net_type == 'resnet101': self.layer = nn.Sequential(models.resnet101(pretrained=pretrained), )
        if net_type == 'resnet152': self.layer = nn.Sequential(models.resnet152(pretrained=pretrained), )
        if net_type == 'resnext101_32x8d': self.layer = nn.Sequential(models.resnext101_32x8d(pretrained=pretrained), )
        if net_type == 'resnext50_32x4d': self.layer = nn.Sequential(models.resnext50_32x4d(pretrained=pretrained), )
        if net_type == 'wide_resnet50_2': self.layer = nn.Sequential(models.wide_resnet50_2(pretrained=pretrained), )
        if net_type == 'wide_resnet101_2': self.layer = nn.Sequential(models.wide_resnet101_2(pretrained=pretrained), )
        if net_type == 'densenet121': self.layer = nn.Sequential(models.densenet121(pretrained=pretrained), )
        if net_type == 'densenet161': self.layer = nn.Sequential(models.densenet161(pretrained=pretrained), )
        if net_type == 'densenet169': self.layer = nn.Sequential(models.densenet169(pretrained=pretrained), )
        if net_type == 'densenet201': self.layer = nn.Sequential(models.densenet201(pretrained=pretrained), )
        if net_type == 'vgg11': self.layer = nn.Sequential(models.vgg11(pretrained=pretrained), )
        if net_type == 'vgg13': self.layer = nn.Sequential(models.vgg13(pretrained=pretrained), )
        if net_type == 'vgg13_bn': self.layer = nn.Sequential(models.vgg13_bn(pretrained=pretrained), )
        if net_type == 'vgg19': self.layer = nn.Sequential(models.vgg19(pretrained=pretrained), )
        if net_type == 'vgg19_bn': self.layer = nn.Sequential(models.vgg19_bn(pretrained=pretrained), )
        if net_type == 'vgg16': self.layer = nn.Sequential(models.vgg16(pretrained=pretrained), )
        if net_type == 'vgg16_bn': self.layer = nn.Sequential(models.vgg16_bn(pretrained=pretrained), )
        if net_type == 'inception_v3': self.layer = nn.Sequential(models.inception_v3(pretrained=pretrained), )
        if net_type == 'mobilenet_v2': self.layer = nn.Sequential(models.mobilenet_v2(pretrained=pretrained), )
        if net_type == 'mobilenet_v3_small': self.layer = nn.Sequential(
            models.mobilenet_v3_small(pretrained=pretrained), )
        if net_type == 'mobilenet_v3_large': self.layer = nn.Sequential(
            models.mobilenet_v3_large(pretrained=pretrained), )
        if net_type == 'shufflenet_v2_x0_5': self.layer = nn.Sequential(
            models.shufflenet_v2_x0_5(pretrained=pretrained), )
        if net_type == 'shufflenet_v2_x1_0': self.layer = nn.Sequential(
            models.shufflenet_v2_x1_0(pretrained=pretrained), )
        if net_type == 'shufflenet_v2_x1_5': self.layer = nn.Sequential(
            models.shufflenet_v2_x1_5(pretrained=pretrained), )
        if net_type == 'shufflenet_v2_x2_0': self.layer = nn.Sequential(
            models.shufflenet_v2_x2_0(pretrained=pretrained), )
        if net_type == 'alexnet': self.layer = nn.Sequential(models.alexnet(pretrained=pretrained), )
        if net_type == 'googlenet': self.layer = nn.Sequential(models.googlenet(pretrained=pretrained), )
        if net_type == 'mnasnet0_5': self.layer = nn.Sequential(models.mnasnet0_5(pretrained=pretrained), )
        if net_type == 'mnasnet1_0': self.layer = nn.Sequential(models.mnasnet1_0(pretrained=pretrained), )
        if net_type == 'mnasnet1_3': self.layer = nn.Sequential(models.mnasnet1_3(pretrained=pretrained), )
        if net_type == 'mnasnet0_75': self.layer = nn.Sequential(models.mnasnet0_75(pretrained=pretrained), )
        if net_type == 'squeezenet1_0': self.layer = nn.Sequential(models.squeezenet1_0(pretrained=pretrained), )
        if net_type == 'squeezenet1_1': self.layer = nn.Sequential(models.squeezenet1_1(pretrained=pretrained), )
        if net_type in ['efficientnet-b0','efficientnet-b1','efficientnet-b2','efficientnet-b3','efficientnet-b4','efficientnet-b5','efficientnet-b6']:
            if pretrained:
                self.layer=nn.Sequential(EfficientNet.from_pretrained(net_type))
            else:
                self.layer=nn.Sequential(EfficientNet.from_name(net_type))
        self.out = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.out(self.layer(x))


