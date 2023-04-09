from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
import torch
from utils.Data_process import five_classes

# 将图像数据转化为numpy型
class MaskToTensor(object):  # 将MaskToTensor定义为可以调用的类
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


# 对图像进行归一化的操作
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])
mask_transform = MaskToTensor()


class RSDataset(Dataset):
    def __init__(self, root=None, mode=None, img_transform=img_transform, mask_transform=mask_transform,
                 sync_transforms=None):
        # 数据相关
        self.class_names = five_classes()  # 图像中所包含的种类
        self.mode = mode
        self.img_transform = img_transform
        self.mask_transform = mask_transform
        self.sync_transform = sync_transforms
        self.sync_img_mask = []

        if mode == "train":
            key_word = 'train_data'
        elif mode == "val":
            key_word = 'val_data'
        else:
            key_word = 'test_data'

        if mode == "src":
            img_dir = os.path.join(root, 'rgb')
            mask_dir = os.path.join(root, 'label')
        else:
            for dirname in os.listdir(root):
                # 进入选定的文件夹
                if dirname == key_word in dirname:
                    break

            # 读取其中的图像数据

            img_dir = os.path.join(root, dirname, 'rgb')
            mask_dir = os.path.join(root, dirname, 'label')

        # 将相应的图像数据进行保存
        for img_filename in os.listdir(img_dir):
            img_mask_pair = (os.path.join(img_dir, img_filename),
                             os.path.join(mask_dir,
                                          img_filename.replace(img_filename[-8:], "label_" + img_filename[-8:])))

            self.sync_img_mask.append(img_mask_pair)

        # print(self.sync_img_mask)
        if (len(self.sync_img_mask)) == 0:
            print("Found 0 data, please check your dataset!")

    def __getitem__(self, index):
        num_class = 6
        ignore_label = 5

        img_path, mask_path = self.sync_img_mask[index]
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # 将图像转化为灰度值

        # 将图像进行相应的裁剪，变换等操作
        if self.sync_transform is not None:
            img, mask = self.sync_transform(img, mask)

        # 将原始图像进行归一化操作
        if self.img_transform is not None:
            img = self.img_transform(img)

        # 将标签图转化为可以操作的形式
        if self.mask_transform is not None:
            mask = self.mask_transform(mask)

        mask[mask >= num_class] = ignore_label
        mask[mask < 0] = ignore_label

        return img, mask

    def __len__(self):
        return len(self.sync_img_mask)

    def classes(self):
        return self.class_names


if __name__ == "__main__":
    pass
    # RSDataset(class_name, root=args.train_data_root, mode='train', sync_transforms=None)
