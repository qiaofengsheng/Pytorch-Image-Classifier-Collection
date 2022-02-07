'''
 ==================板块功能描述====================
           @Time     :2022/2/7   15:55
           @Author   : qiaofengsheng
           @File     :pruning_model.py
           @Software :PyCharm
           @description:模型剪枝、量化压缩
           支持FPGMPruner,L1FilterPruner,L2FilterPruner裁剪方式
           其他裁剪方式待完善
 ================================================
 '''

# 微调模型训练函数
import sys
import os

import torch

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
import argparse
from nni.compression.pytorch import ModelSpeedup
from model.dataset.dataset import *
from model.loss.loss_fun import Loss
from model.optimizer.optim import Optimizer
from model.utils.utils import *
from model.net.net import *
from nni.algorithms.compression.pytorch.pruning import (
    FPGMPruner,
    L1FilterPruner,
    L2FilterPruner,
    LevelPruner,
    SlimPruner,
    AGPPruner,
    TaylorFOWeightFilterPruner,
    ActivationMeanRankFilterPruner,
    ActivationAPoZRankFilterPruner
)

prune_save_path = 'prune_model/prune_checkpoints'
if not os.path.exists(prune_save_path):
    os.makedirs(prune_save_path)

parse = argparse.ArgumentParser('pruning model')
parse.add_argument('--weight_path', type=str, default=None, help='已训练好的模型权重地址')
parse.add_argument('--prune_type', type=str, default='l1filter', help='修剪模型的方式，支持：l1filter,l2filter,fpgm')
parse.add_argument('--sparsity', type=float, default=0.5, help='模型稀疏化比例')
parse.add_argument('--op_names', type=list, default=None, help='指定修剪哪些layer名字，默认修剪全部的Conv2d,输入为样例：["conv1","conv2"]')
parse.add_argument('--finetune_epoches', type=int, default=10, help='微调模型的轮次数')
parse.add_argument('--dummy_input', type=str, required=True, help='输入模型的形状，例如：(10,3,128,128)')


def trainer(model, train_loader, optimizer, criterion, epoch, device):
    model = model.to(device)
    model.train()
    for idx, (image_data, target) in enumerate(train_loader):
        image_data, target = image_data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(image_data)
        train_loss = criterion(output, target)
        train_loss.backward()
        optimizer.step()
        if idx % 20 == 0:
            print(f'第{epoch}轮--第{idx}批次--train_loss : {train_loss.item()}')


# 微调模型评估函数
def evaluator(model, test_loader, criterion, epoch, device):
    model = model.to(device)
    index = 0
    test_loss = 0
    acc = 0
    model.eval()
    for image_data, target in test_loader:
        image_data, target = image_data.to(device), target.to(device)
        output = model(image_data)
        test_loss += criterion(output, target)
        pred = output.argmax(dim=1)
        acc += torch.mean(torch.eq(pred, target).float()).item()
        index += 1
    test_loss /= index
    acc /= index
    print(f'第{epoch}轮--Average test_loss : {test_loss} -- Average Accuracy : {acc}')
    return acc


def prune_tools(args, model, train_loader, test_loader, criterion, optimizer, device):
    model.load_state_dict(torch.load(args.weight_path))
    model = model.to(device)
    print(model)
    if args.op_names is None:
        config_list = [{
            'sparsity': args.sparsity,
            'op_types': ['Conv2d']
        }]
    else:
        config_list = [{
            'sparsity': args.sparsity,
            'op_type': ['Conv2d'],
            'op_names': args.op_names
        }]
    prune_type = {
        'l1filter': L1FilterPruner,
        'l2filter': L2FilterPruner,
        'fpgm': FPGMPruner
    }
    # 裁剪模型
    pruner = prune_type[args.prune_type](model, config_list)
    pruner.compress()

    # 导出稀疏模型和掩码模型
    pruner.export_model(model_path=os.path.join(prune_save_path, 'sparsity_model.pth'),
                        mask_path=os.path.join(prune_save_path, 'mask_model.pth'))

    # 打开新模型
    pruner._unwrap_model()
    # 模型加速
    dummy_input = args.dummy_input.split(',')
    n, c, h, w = int(dummy_input[0][1:]), int(dummy_input[1]), int(dummy_input[2]), int(dummy_input[3][:-1])
    m_speedup = ModelSpeedup(model, dummy_input=torch.randn(20, 3, 128, 128).to(device),
                             masks_file=r'C:\Users\Administrator\Desktop\Pytorch-Image-Classifier-Collection\prune_model\prune_checkpoints\mask_model.pth')
    m_speedup.speedup_model()

    # 微调模型
    best_acc = 0
    for epoch in range(1, args.finetune_epoches + 1):
        trainer(model, train_loader, optimizer, criterion, epoch, device)
        acc = evaluator(model, test_loader, criterion, epoch, device)
        if acc > best_acc:
            torch.save(model, os.path.join(prune_save_path, 'pruned_model.pth'))
            print('successfully save pruned_model weights!')
            best_acc = acc
        else:
            continue
    print(f'微调后的模型准确率为 : {best_acc * 100}%')


if __name__ == '__main__':
    args = parse.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = load_config_util('config/config.yaml')
    dataset = ClassDataset(config['data_dir'], config)
    train_dataset, test_dataset = random_split(dataset,
                                               [int(len(dataset) * config['train_rate']),
                                                len(dataset) - int(
                                                    len(dataset) * config['train_rate'])]
                                               )
    train_data_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=True)
    model = ClassifierNet(config['net_type'], len(config['class_names']), False)
    model.load_state_dict(torch.load(args.weight_path))
    criterion = Loss(config['loss_type']).get_loss_fun()
    optimizer = Optimizer(model, config['optimizer']).get_optimizer()
    prune_tools(args, model, train_data_loader, test_data_loader, criterion, optimizer, device)
