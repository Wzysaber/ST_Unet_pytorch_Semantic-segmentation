from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import numpy as np
import data.sync_transforms

from tqdm import tqdm
from data.dataset import RSDataset
from torch.autograd import Variable
from prettytable import PrettyTable
from Parameter import average_meter, metric


def close_optimizer(args, model):
    # 使用相应的优化器
    if args.optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(model.parameters(),
                                         lr=args.base_lr,
                                         weight_decay=args.weight_decay)
    if args.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.base_lr)

    if args.optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(),
                                    lr=args.base_lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)

    return optimizer


def data_set(args):
    # 对载入图像进行数据增强
    resize_scale_range = [float(scale) for scale in args.resize_scale_range.split(',')]  # 0.5 2.0

    sync_transform = data.sync_transforms.Compose([
        data.sync_transforms.RandomScale(args.base_size, args.crop_size, resize_scale_range),
        data.sync_transforms.RandomFlip(args.flip_ratio)
    ])

    # 数据集的载入和相应参数
    train_dataset = RSDataset(root=args.train_data_root, mode='src', sync_transforms=sync_transform)  # 加载数据集

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.train_batch_size,
                              num_workers=args.num_workers,
                              shuffle=True,
                              drop_last=True)

    # print('class names {}.'.format(train_loader.class_names))
    # print('Number samples {}.'.format(len(train_loader)))  # 将模型的种类数和名称进行打印

    # 实现相应验证集
    if not args.no_val:
        val_dataset = RSDataset(root=args.val_data_root, mode='src', sync_transforms=None)
        val_loader = DataLoader(dataset=val_dataset,
                                batch_size=args.val_batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                drop_last=True)

    return train_loader, train_dataset, val_loader, val_dataset


def training(args, num_classes, model, optimizer, train_dataset, train_loader, criterion1, criterion2, device, epoch):
    model.train()  # 把module设成训练模式，对Dropout和BatchNorm有影响

    train_loss = average_meter.AverageMeter()

    # “Poly”衰减策略
    max_iter = args.total_epochs * len(train_loader)
    curr_iter = epoch * len(train_loader)  # 训练的数量
    lr = args.base_lr * (1 - float(curr_iter) / max_iter) ** 0.9  # 自己定义的学习率

    # 建立比较的矩阵16X16的格式，
    conf_mat = np.zeros((5, 5)).astype(np.int64)

    tbar = tqdm(train_loader)  # 可视化显示数据的迭代

    # 将训练集里面的数据进行相应的遍历
    for index, data in enumerate(tbar):
        # assert data[0].size()[2:] == data[1].size()[1:]
        # data = self.mixup_transform(data, epoch)

        # 加载dataload中的图片
        imgs = Variable(data[0]).to(device)
        masks = Variable(data[1]).to(device)

        # 引入参数
        outputs = model(imgs)
        # torch.max(tensor, dim)：指定维度上最大的数，返回tensor和下标
        _, preds = torch.max(outputs, 1)  # 加_,则返回一行中最大数的位置。
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)  # 将数据提取出来

        loss1 = criterion1(outputs, masks)
        loss2 = criterion2(outputs, masks, softmax=True)

        loss = 0.5 * loss1 + 0.5 * loss2

        train_loss.update(loss, args.train_batch_size)
        # writer.add_scalar('train_loss', train_loss.avg, curr_iter)

        optimizer.zero_grad()  # zero_grad()梯度清0
        loss.backward()
        optimizer.step()

        # 将相应的数据进行打印
        tbar.set_description('epoch {}, training loss {}, with learning rate {}.'.format(
            epoch, train_loss.val, lr
        ))

        masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)  # 将数据提取出来

        # 将相应的数据存储在矩阵方阵中
        conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                            label=masks.flatten(),
                                            num_classes=num_classes)

    # 评价参数
    train_acc, train_acc_per_class, train_pre, train_IoU, train_mean_IoU, train_kappa, train_F1_score, train_recall = metric.evaluate(
        conf_mat)

    table = PrettyTable(["序号", "名称", "acc", "IOu"])

    # 打印参数
    for i in range(5):
        table.add_row([i, train_dataset.class_names[i], train_acc_per_class[i], train_IoU[i]])
    print(table)

    print("F1_score:", train_F1_score)
    print("train_mean_IoU:", train_mean_IoU)

    print("\ntrain_acc(OA):", train_acc)
    print("kappa:", train_kappa)
    print(" ")

    return train_acc
