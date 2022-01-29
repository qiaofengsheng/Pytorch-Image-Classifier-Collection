'''
 _*_coding:utf-8 _*_
 @Time     :2022/1/29   10:52
 @Author   : qiaofengsheng
 @File     :infer.py
 @Software :PyCharm
 '''
import os

from PIL import Image, ImageDraw, ImageFont
import cv2
import torch
from model.utils import utils
from torchvision import transforms
from model.net.net import *
import argparse

parse = argparse.ArgumentParser('infer models')
parse.add_argument('demo', type=str, help='推理类型支持：image/video/camera')
parse.add_argument('--weights_path', type=str, default='', help='模型权重路径')
parse.add_argument('--image_path', type=str, default='', help='图片存放路径')
parse.add_argument('--video_path', type=str, default='', help='视频路径')
parse.add_argument('--camera_id', type=int, default=0, help='摄像头id')


class ModelInfer:
    def __init__(self, config, weights_path):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.net = ClassifierNet(self.config['net_type'], len(self.config['class_names']),
                                 False).to(self.device)
        if weights_path is not None:
            if os.path.exists(weights_path):
                self.net.load_state_dict(torch.load(weights_path))
                print('successfully loading model weights!')
            else:
                print('no loading model weights!')
        else:
            print('please input weights_path!')
            exit(0)
        self.net.eval()

    def image_infer(self, image_path):
        image = Image.open(image_path)
        image_data = utils.keep_shape_resize(image, self.config['image_size'])
        image_data = self.transform(image_data)
        image_data = torch.unsqueeze(image_data, dim=0).to(self.device)
        out = self.net(image_data)
        out = torch.argmax(out)
        result = self.config['class_names'][int(out)]
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(r"C:\Windows\Fonts\BRITANIC.TTF", 35)
        draw.text((10, 10), result, font=font, fill='red')
        image.show()

    def video_infer(self, video_path):
        cap = cv2.VideoCapture(video_path)
        while True:
            _, frame = cap.read()
            if _:
                image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_data = Image.fromarray(image_data)
                image_data = utils.keep_shape_resize(image_data, self.config['image_size'])
                image_data = self.transform(image_data)
                image_data = torch.unsqueeze(image_data, dim=0).to(self.device)
                out = self.net(image_data)
                out = torch.argmax(out)
                result = self.config['class_names'][int(out)]
                cv2.putText(frame, result, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(24) & 0XFF == ord('q'):
                    break
            else:
                break

    def camera_infer(self, camera_id):
        cap = cv2.VideoCapture(camera_id)
        while True:
            _, frame = cap.read()
            h,w,c=frame.shape
            if _:
                image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_data = Image.fromarray(image_data)
                image_data = utils.keep_shape_resize(image_data, self.config['image_size'])
                image_data = self.transform(image_data)
                image_data = torch.unsqueeze(image_data, dim=0).to(self.device)
                out = self.net(image_data)
                out = torch.argmax(out)
                result = self.config['class_names'][int(out)]
                cv2.putText(frame, result, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(24) & 0XFF == ord('q'):
                    break
            else:
                break


if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util('config/config.yaml')
    model = ModelInfer(config, args.weights_path)
    if args.demo == 'image':
        model.image_infer(args.image_path)
    elif args.demo == 'video':
        model.video_infer(args.video_path)
    elif args.demo == 'camera':
        model.camera_infer(args.camera_id)
    else:
        exit(0)
