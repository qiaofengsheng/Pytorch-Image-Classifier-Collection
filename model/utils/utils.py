'''
 _*_coding:utf-8 _*_
 @Time     :2022/1/28   19:58
 @Author   : qiaofengsheng
 @File     :utils.py
 @Software :PyCharm
 '''
import torch
import yaml
from PIL import Image
from torch.nn.functional import one_hot


def load_config_util(config_path):
    config_file = open(config_path, 'r', encoding='utf-8')
    config_data = yaml.load(config_file)
    return config_data


def keep_shape_resize(frame, size=256):
    w, h = frame.size
    temp = max(w, h)
    mask = Image.new('RGB', (temp, temp), (0, 0, 0))
    if w >= h:
        position = (0, (w - h) // 2)
    else:
        position = ((h - w) // 2, 0)
    mask.paste(frame, position)
    mask = mask.resize((size, size))
    return mask


def label_one_hot(label):
    return one_hot(torch.tensor(label))
