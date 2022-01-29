'''
 _*_coding:utf-8 _*_
 @Time     :2022/1/28   19:05
 @Author   : qiaofengsheng
 @File     :loss_fun.py
 @Software :PyCharm
 '''

from torch import nn


class Loss():
    def __init__(self, loss_type='mse'):
        self.loss_fun = nn.MSELoss()
        if loss_type == 'mse':
            self.loss_fun = nn.MSELoss()
            self.loss_fun = nn.L1Loss()
            self.loss_fun = nn.SmoothL1Loss()
            self.loss_fun = nn.BCELoss()
            self.loss_fun = nn.CrossEntropyLoss()

    def __call__(self):
        pass
