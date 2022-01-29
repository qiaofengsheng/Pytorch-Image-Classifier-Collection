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


class Net(nn.Module):
    def __init__(self, net_name='resnet18', num_classes=10, pretrained=False):
        super(Net, self).__init__()
        self.layer = nn.Sequential(models.resnet18(pretrained=pretrained))
        if net_name == 'resnet18': self.layer = nn.Sequential(models.resnet18(pretrained=pretrained), )
        if net_name == 'resnet34': self.layer = nn.Sequential(models.resnet34(pretrained=pretrained), )
        if net_name == 'resnet50': self.layer = nn.Sequential(models.resnet50(pretrained=pretrained), )
        if net_name == 'resnet101': self.layer = nn.Sequential(models.resnet101(pretrained=pretrained), )
        if net_name == 'resnet152': self.layer = nn.Sequential(models.resnet152(pretrained=pretrained), )
        if net_name == 'resnext101_32x8d': self.layer = nn.Sequential(models.resnext101_32x8d(pretrained=pretrained), )
        if net_name == 'resnext50_32x4d': self.layer = nn.Sequential(models.resnext50_32x4d(pretrained=pretrained), )
        if net_name == 'wide_resnet50_2': self.layer = nn.Sequential(models.wide_resnet50_2(pretrained=pretrained), )
        if net_name == 'wide_resnet101_2': self.layer = nn.Sequential(models.wide_resnet101_2(pretrained=pretrained), )
        if net_name == 'densenet121': self.layer = nn.Sequential(models.densenet121(pretrained=pretrained), )
        if net_name == 'densenet161': self.layer = nn.Sequential(models.densenet161(pretrained=pretrained), )
        if net_name == 'densenet169': self.layer = nn.Sequential(models.densenet169(pretrained=pretrained), )
        if net_name == 'densenet201': self.layer = nn.Sequential(models.densenet201(pretrained=pretrained), )
        if net_name == 'vgg11': self.layer = nn.Sequential(models.vgg11(pretrained=pretrained), )
        if net_name == 'vgg13': self.layer = nn.Sequential(models.vgg13(pretrained=pretrained), )
        if net_name == 'vgg13_bn': self.layer = nn.Sequential(models.vgg13_bn(pretrained=pretrained), )
        if net_name == 'vgg19': self.layer = nn.Sequential(models.vgg19(pretrained=pretrained), )
        if net_name == 'vgg19_bn': self.layer = nn.Sequential(models.vgg19_bn(pretrained=pretrained), )
        if net_name == 'vgg16': self.layer = nn.Sequential(models.vgg16(pretrained=pretrained), )
        if net_name == 'vgg16_bn': self.layer = nn.Sequential(models.vgg16_bn(pretrained=pretrained), )
        if net_name == 'inception_v3': self.layer = nn.Sequential(models.inception_v3(pretrained=pretrained), )
        if net_name == 'mobilenet_v2': self.layer = nn.Sequential(models.mobilenet_v2(pretrained=pretrained), )
        if net_name == 'mobilenet_v3_small': self.layer = nn.Sequential(
            models.mobilenet_v3_small(pretrained=pretrained), )
        if net_name == 'mobilenet_v3_large': self.layer = nn.Sequential(
            models.mobilenet_v3_large(pretrained=pretrained), )
        if net_name == 'shufflenet_v2_x0_5': self.layer = nn.Sequential(
            models.shufflenet_v2_x0_5(pretrained=pretrained), )
        if net_name == 'shufflenet_v2_x1_0': self.layer = nn.Sequential(
            models.shufflenet_v2_x1_0(pretrained=pretrained), )
        if net_name == 'shufflenet_v2_x1_5': self.layer = nn.Sequential(
            models.shufflenet_v2_x1_5(pretrained=pretrained), )
        if net_name == 'shufflenet_v2_x2_0': self.layer = nn.Sequential(
            models.shufflenet_v2_x2_0(pretrained=pretrained), )
        if net_name == 'alexnet': self.layer = nn.Sequential(models.alexnet(pretrained=pretrained), )
        if net_name == 'googlenet': self.layer = nn.Sequential(models.googlenet(pretrained=pretrained), )
        if net_name == 'mnasnet0_5': self.layer = nn.Sequential(models.mnasnet0_5(pretrained=pretrained), )
        if net_name == 'mnasnet1_0': self.layer = nn.Sequential(models.mnasnet1_0(pretrained=pretrained), )
        if net_name == 'mnasnet1_3': self.layer = nn.Sequential(models.mnasnet1_3(pretrained=pretrained), )
        if net_name == 'mnasnet0_75': self.layer = nn.Sequential(models.mnasnet0_75(pretrained=pretrained), )
        if net_name == 'squeezenet1_0': self.layer = nn.Sequential(models.squeezenet1_0(pretrained=pretrained), )
        if net_name == 'squeezenet1_1': self.layer = nn.Sequential(models.squeezenet1_1(pretrained=pretrained), )

        self.out = nn.Linear(1000, num_classes)

    def forward(self, x):
        return self.out(self.layer(x))


if __name__ == '__main__':
    net = Net('resnet50', 5)
    x = torch.randn(1, 3, 520, 520)
    print(net(x).shape)
