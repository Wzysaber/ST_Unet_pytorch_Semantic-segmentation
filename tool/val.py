import numpy as np
import torch
import os

from tqdm import tqdm
from torch.autograd import Variable
from Parameter import metric
from prettytable import PrettyTable


def validating(args, num_classes, model, optimizer, train_dataset, val_loader, device, epoch):
    model.eval()  # 把module 设成预测模式，对Dropout和BatchNorm有影响

    # 构建矩阵方阵
    conf_mat = np.zeros((5, 5)).astype(np.int64)
    # 加载相应的数据集
    tbar = tqdm(val_loader)

    # 对数据进行遍历
    for index, data in enumerate(tbar):
        # assert data[0].size()[2:] == data[1].size()[1:]

        # 将相应的数据提取出来
        imgs = Variable(data[0]).to(device)
        masks = Variable(data[1]).to(device)

        optimizer.zero_grad()  # 梯度清0
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)  # 返回最大值的值，不是像素的值则为1

        # 将相应的参数进行提取
        preds = preds.data.cpu().numpy().squeeze().astype(np.uint8)
        masks = masks.data.cpu().numpy().squeeze().astype(np.uint8)

        conf_mat += metric.confusion_matrix(pred=preds.flatten(),
                                            label=masks.flatten(),
                                            num_classes=num_classes)

    # 打印相应的数据
    val_acc, val_acc_per_class, val_pre, val_IoU, val_mean_IoU, val_kappa, val_F1_score, val_recall = metric.evaluate(
        conf_mat)

    model_name = 'epoch_%d_miou_%.2f_F1_%.2f' % (epoch, val_mean_IoU, val_F1_score)

    # 保存相应训练中最好的模型
    if val_mean_IoU > args.best_miou:
        if args.save_file:
            torch.save(model.state_dict(), os.path.join(args.directory, model_name + '.pth'))
        args.best_miou = val_mean_IoU

    table = PrettyTable(["序号", "名称", "acc", "IoU"])

    for i in range(5):
        table.add_row([i, train_dataset.class_names[i], val_acc_per_class[i], val_IoU[i]])
    print(table)
    print("val_F1_score:", val_F1_score)
    print("val_mean_IoU:", val_mean_IoU)
    print("val_acc:", val_acc)
    print("best_miou:", args.best_miou)
