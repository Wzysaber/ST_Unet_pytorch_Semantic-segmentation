import torch
import torch.nn as nn

from configs.configs import parse_args
from model.deeplabv3_version_1.deeplabv3 import DeepLabV3
from model.Unet.Unet import Unet
# from model.ST_Unet.vit_seg_modeling import VisionTransformer
# from model.ST_Unet.vit_seg_configs import get_r50_b16_config
from model.SwinUnet.vision_transformer import SwinUnet
from model.TransUnet.vit_seg_configs import get_r50_b16_config
from model.TransUnet.vit_seg_modeling import VisionTransformer
from model.Swin_Transformer.SwinT import SwinTransformerV2

from tool.train import close_optimizer
from tool.train import data_set
from tool.train import training
from tool.val import validating

from utils.Loss import DiceLoss
from utils.Data_process import Print_data
from utils.Data_process import Creat_LineGraph

# 忽略相应的警告
import warnings

warnings.filterwarnings("ignore")

# 清除pytorch无用缓存
import gc

gc.collect()
torch.cuda.empty_cache()

# # 设置GPU的序列号
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 设置采用的GPU序号


def main():
    # # 所以，当这个参数设置为True时，启动算法的前期会比较慢，但算法跑起来以后会非常快
    torch.backends.cudnn.benchmark = True

    # 导入配置
    args = parse_args()

    # 加载训练和验证数据集
    train_loader = data_set(args)[0]
    train_dataset = data_set(args)[1]

    val_loader = data_set(args)[2]

    # 训练的相关配置
    device = torch.device("cuda:2")

    # 加载模型
    if args.model == "Unet":
        model = Unet(num_classes=6).to(device)
    elif args.model == "ST-Unet":
        config_vit = get_r50_b16_config()
        model = VisionTransformer(config_vit, img_size=256, num_classes=6).to(device)
    elif args.model == "deeplabv3+":
        model = DeepLabV3(num_classes=6).to(device)
    elif args.model == "SwinUnet":
        model = SwinUnet(num_classes=6).to(device)
    elif args.model == "TransUnet":
        config_vit = get_r50_b16_config()
        model = VisionTransformer(config_vit, img_size=256, num_classes=6).to(device)
    elif args.model == "Swin_Transformer":
        model = SwinTransformerV2().to(device)

    # 判断是否有训练好的模型
    if args.resume_path:
        state_dict = torch.load('.pth')
        model.load_state_dict(state_dict, state_dict=False)

    # 损失函数
    criterion1 = nn.CrossEntropyLoss().to(device)
    criterion2 = DiceLoss(6).to(device)

    # 优化器选择
    optimizer = close_optimizer(args, model).to(device)

    # 将相应的参数进行打印
    Print_data(args.dataset_name, train_dataset.class_names,
               train_dataset, args.optimizer_name, args.model, args.total_epochs)

    # 训练及验证
    traincd_Data = []
    for epoch in range(args.start_epoch, args.total_epochs):
        ACC = training(args, 6, model, optimizer, train_dataset, train_loader, criterion1, criterion2, device,
                       epoch)  # 对模型进行训练zzzz
        validating(args, 6, model, optimizer, train_dataset, val_loader, device, epoch)  # 对模型进行验证
        traincd_Data.append(ACC)
        print(" ")
    Creat_LineGraph(traincd_Data)  # 绘制相应曲线图


if __name__ == "__main__":
    main()
