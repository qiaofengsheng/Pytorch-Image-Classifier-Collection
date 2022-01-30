'''
 _*_coding:utf-8 _*_
 @Time     :2022/1/30   10:28
 @Author   : qiaofengsheng
 @File     :pytorch_onnx_infer.py
 @Software :PyCharm
 '''
import os
import sys

import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import cv2
import onnxruntime
import argparse
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
import torch
from model.utils import utils

parse = argparse.ArgumentParser(description='onnx model infer!')
parse.add_argument('demo', type=str, help='推理类型支持：image/video/camera')
parse.add_argument('--config_path', type=str, help='配置文件存放地址')
parse.add_argument('--onnx_path', type=str, default=None, help='onnx包存放路径')
parse.add_argument('--image_path', type=str, default='', help='图片存放路径')
parse.add_argument('--video_path', type=str, default='', help='视频路径')
parse.add_argument('--camera_id', type=int, default=0, help='摄像头id')
parse.add_argument('--device', type=str, default='cpu', help='默认设备cpu （暂未完善GPU代码）')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def onnx_infer_image(args, config):
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(args.image_path)
    image_data = utils.keep_shape_resize(image, config['image_size'])
    image_data = transform(image_data)
    image_data = torch.unsqueeze(image_data, dim=0)
    if args.device == 'cpu':
        ort_input = {ort_session.get_inputs()[0].name: to_numpy(image_data)}
        ort_out = ort_session.run(None, ort_input)
        out = np.argmax(ort_out[0], axis=1)
        result = config['class_names'][int(out)]
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(r"C:\Windows\Fonts\BRITANIC.TTF", 35)
        draw.text((10, 10), result, font=font, fill='red')
        image.show()
    elif args.device == 'cuda':
        pass
    else:
        exit(0)


def onnx_infer_video(args, config):
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    transform = transforms.Compose([transforms.ToTensor()])
    cap = cv2.VideoCapture(args.video_path)
    while True:
        _, frame = cap.read()
        if _:
            image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_data = Image.fromarray(image_data)
            image_data = utils.keep_shape_resize(image_data, config['image_size'])
            image_data = transform(image_data)
            image_data = torch.unsqueeze(image_data, dim=0)
            if args.device == 'cpu':
                ort_input = {ort_session.get_inputs()[0].name: to_numpy(image_data)}
                ort_out = ort_session.run(None, ort_input)
                out = np.argmax(ort_out[0], axis=1)
                result = config['class_names'][int(out)]
                cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(24) & 0XFF == ord('q'):
                    break
            elif args.device == 'cuda':
                pass
            else:
                exit(0)
        else:
            exit(0)


def onnx_infer_camera(args, config):
    ort_session = onnxruntime.InferenceSession(args.onnx_path)
    transform = transforms.Compose([transforms.ToTensor()])
    cap = cv2.VideoCapture(args.camera_id)
    while True:
        _, frame = cap.read()
        if _:
            image_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_data = Image.fromarray(image_data)
            image_data = utils.keep_shape_resize(image_data, config['image_size'])
            image_data = transform(image_data)
            image_data = torch.unsqueeze(image_data, dim=0)
            if args.device == 'cpu':
                ort_input = {ort_session.get_inputs()[0].name: to_numpy(image_data)}
                ort_out = ort_session.run(None, ort_input)
                out = np.argmax(ort_out[0], axis=1)
                result = config['class_names'][int(out)]
                cv2.putText(frame, result, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=2)
                cv2.imshow('frame', frame)
                if cv2.waitKey(24) & 0XFF == ord('q'):
                    break
            elif args.device == 'cuda':
                pass
            else:
                exit(0)
        else:
            exit(0)


if __name__ == '__main__':
    args = parse.parse_args()
    config = utils.load_config_util(args.config_path)
    if args.demo == 'image':
        onnx_infer_image(args, config)
    elif args.demo == 'video':
        onnx_infer_video(args, config)
    elif args.demo == 'camera':
        onnx_infer_camera(args, config)
    else:
        exit(0)
