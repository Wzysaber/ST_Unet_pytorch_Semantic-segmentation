import os
import shutil
import matplotlib.pyplot as plt

# 自定义类别
def fifteen_classes():
    return ['其他类别',
            '水田',
            '水浇地',
            '旱耕地',
            '园地',
            '乔木林地',
            '灌木林地',
            '天然草地',
            '人工草地',
            '工业用地',
            '城市住宅',
            '村镇住宅',
            '交通运输',
            '河流',
            '湖泊',
            '坑塘']


def five_classes():
    return [
        '不透明表面',
        '建筑',
        '灌木',
        '树',
        '车',
    ]


def Print_data(dataset_name, class_name, train_dataset_len, optimizer_name, model, total_epochs):
    print('\ndataset:', dataset_name)
    print('classification:', class_name)
    print('Number samples {}.'.format(len(train_dataset_len)))  # 将模型的种类数和名称进行打印
    print('\noptimizer:', optimizer_name)
    print('model:', model)
    print('epoch:', total_epochs)
    print("\nOK!,everything is fine,let's start training!\n")


def Creat_LineGraph(traincd_line):
    x = range(len(traincd_line))
    y = traincd_line
    plt.plot(x, y, color="g", label="train cd H_acc", linewidth=0.3, marker=',')
    plt.xlabel('Epoch')
    plt.ylabel('Acc Value')
    plt.show()
