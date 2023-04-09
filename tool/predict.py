import torch
import torchvision.transforms as T
import numpy as np
import cv2
from PIL import Image
from model.Unet.Unet import Unet
import os

from utils.palette import colorize_mask
from Parameter import metric
from prettytable import PrettyTable


import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置采用的GPU序号


# 定义预测函数
def predict(model, image_path, Gray_label_path):
    """
    对输入图像进行预测，返回预测结果。

    Args:
    model (nn.Module): PyTorch模型实例
    image_path (str): 输入图像路径

    Returns:
    预测结果的(N, H, W)的numpy数组
    """
    # 加载图像并做相应预处理
    img = Image.open(image_path).convert('RGB')

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = transform(img).to(device)
    img = img.unsqueeze(0)  # 增加batch维

    Gray_label = Image.open(Gray_label_path).convert('L')
    mask = torch.from_numpy(np.array(Gray_label, dtype=np.int8)).long().numpy()

    # 对输入图像进行预测

    output = model(img)
    # pred = output.argmax(dim=1)  # 取最大值的索引
    _, pred = torch.max(output, 1)  # 加_,则返回一行中最大数的位置。

    # 转为numpy数组并去掉batch维
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)  # 将数据提取出来

    return pred, mask


def Check(pred, mask):
    conf_mat = np.zeros((5, 5)).astype(np.int64)
    conf_mat += metric.confusion_matrix(pred=pred.flatten(),
                                        label=mask.flatten(),
                                        num_classes=6)

    acc, acc_per_class, pre, IoU, mean_IoU, kappa, F1_score, val_recall = metric.evaluate(conf_mat)

    print("Mean_IoU:", mean_IoU)
    print("OA:", acc)


if __name__ == '__main__':
    # 加载相应参数
    device = torch.device("cuda:3")

    image_path = "/home/students/master/2022/wangzy/dataset/Vaihingen/predict/Cut/rgb/top_mosaic_09cm_area1_0017.jpg"
    RGB_label_path = "/home/students/master/2022/wangzy/dataset/Vaihingen/predict/Cut/label/top_mosaic_09cm_area1_label_0017.jpg"
    Gray_label_path = "/home/students/master/2022/wangzy/dataset/Vaihingen/predict/Cut/Gray_label/top_mosaic_09cm_area1_label_0017.jpg"

    model_path = "/home/students/master/2022/wangzy/PyCharm-Remote/ST_Unet_test/weight/Vaihingen/Unet/03-21-22:41:16/epoch_84_miou_0.70_F1_0.82.pth"  # 导入网络的参数

    # 加载原始标签
    image = cv2.imread(RGB_label_path)
    B, G, R = cv2.split(image)
    image = cv2.merge((R, G, B))

    # 加载模型
    model = Unet(num_classes=6)

    state_dict = torch.load(model_path)
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     print(k)
    #     name = k[7:]
    #     new_state_dict[name] = v
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)

    # 预测图像
    pred, mask = predict(model, image_path, Gray_label_path)
    overlap = colorize_mask(pred)

    # 查看评价指标
    Check(pred, mask)

    # 可视化预测结果
    plt.title("predict")
    plt.imshow(overlap)
    plt.show()

    plt.title("label")
    plt.imshow(image)
    plt.show()
