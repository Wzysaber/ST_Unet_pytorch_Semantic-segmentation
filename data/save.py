import os
import json
import numpy as np
from configs.configs import parse_args
from PIL import Image
from utils.palette import colorize_mask
from torchvision import transforms

args = parse_args()


# 保存相应的工作参数
def save_work():
    directory = "work_dirs/%s/%s/%s/%s/" % (args.dataset_name, args.model, args.backbone, args.experiment_start_time)
    args.directory = directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    config_file = os.path.join(directory, 'config.json')

    # 将相应参数转换为json格式，进行文本保存
    with open(config_file, 'w') as file:
        json.dump(vars(args), file, indent=4)

    if args.use_cuda:
        print('Numbers of GPUs:', args.num_GPUs)
    else:
        print("Using CPU")


# 归一化操作
# zip的作用将元素打包成为元组
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


resore_transform = transforms.Compose([
    DeNormalize([.485, .456, .406], [.229, .224, .225]),  # 对相应的数据进行归一化操作
    transforms.ToPILImage()  # 将图片变化为可以查看的形式
])


def save_pic(score, data, preds, save_path, epoch, index):
    val_visual = []
    # 将相应的图片进行保存到文件夹
    for i in range(score.shape[0]):

        num_score = np.sum(score[i] > 0.9)

        if num_score > 0.9 * (512 * 512):
            # 将图片进行归一化操作
            # 提取原始图像后进行操作
            img_pil = resore_transform(data[0][i])

            # 将图片转化为灰度图片
            # 这个是我的预测图像
            preds_pil = Image.fromarray(preds[i].astype(np.uint8)).convert('L')

            # 将预测图片转化为RGB
            pred_vis_pil = colorize_mask(preds[i])

            # 将图片转化为RGB
            gt_vis_pil = colorize_mask(data[1][i].numpy())

            # 将相应的数据包装起来
            dir_list = ['rgb', 'label', 'vis_label', 'gt']
            rgb_save_path = os.path.join(save_path, dir_list[0], str(epoch))
            label_save_path = os.path.join(save_path, dir_list[1], str(epoch))
            vis_save_path = os.path.join(save_path, dir_list[2], str(epoch))
            gt_save_path = os.path.join(save_path, dir_list[3], str(epoch))

            path_list = [rgb_save_path, label_save_path, vis_save_path, gt_save_path]

            # 创建相应的地址位置
            for path in range(4):
                if not os.path.exists(path_list[path]):
                    os.makedirs(path_list[path])

            # 将相应的地址位进行保存
            img_pil.save(os.path.join(path_list[0], 'img_batch_%d_%d.jpg' % (index, i)))
            preds_pil.save(os.path.join(path_list[1], 'label_%d_%d.png' % (index, i)))
            pred_vis_pil.save(os.path.join(path_list[2], 'vis_%d_%d.png' % (index, i)))
            gt_vis_pil.save(os.path.join(path_list[3], 'gt_%d_%d.png' % (index, i)))
