'''
 _*_coding:utf-8 _*_
 @Time     :2022/1/28   19:00
 @Author   : qiaofengsheng
 @File     :dataset.py
 @Software :PyCharm
 '''
import os

from PIL import Image
from torch.utils.data import *
from model.utils import utils
from torchvision import transforms


class ClassDataset(Dataset):
    def __init__(self, data_dir, config):
        self.config = config
        self.transform = transforms.Compose([
            transforms.RandomRotation(60),
            transforms.ToTensor()
        ])
        self.dataset = []
        class_dirs = os.listdir(data_dir)
        for class_dir in class_dirs:
            image_names = os.listdir(os.path.join(data_dir, class_dir))
            for image_name in image_names:
                self.dataset.append(
                    [os.path.join(data_dir, class_dir, image_name),
                     int(config['class_names'].index(class_dir))])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        image_path, image_label = data
        image = Image.open(image_path)
        image = utils.keep_shape_resize(image, self.config['image_size'])
        return self.transform(image), image_label
