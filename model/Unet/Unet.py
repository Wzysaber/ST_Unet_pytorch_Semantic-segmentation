import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class Unet(nn.Module):
    def __init__(self, num_classes):
        super(Unet, self).__init__()
        self.stage_1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.stage_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.stage_3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.stage_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        self.stage_5 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
        )

        self.upsample_4 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=1)
        )
        self.upsample_3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1)
        )
        self.upsample_2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1)
        )
        self.upsample_1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1)
        )

        self.stage_up_4 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.stage_up_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.stage_up_2 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.stage_up_1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.final = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.float()
        # 下采样过程
        stage_1 = self.stage_1(x)
        stage_2 = self.stage_2(stage_1)
        stage_3 = self.stage_3(stage_2)
        stage_4 = self.stage_4(stage_3)
        stage_5 = self.stage_5(stage_4)


        # 上采样和合并stage_4
        up_4 = self.upsample_4(stage_5)

        up_4_conv = torch.cat([up_4, stage_4], dim=1)
        up_4_conv = self.stage_up_4(up_4_conv)

        # 上采样和合并stage_3
        up_3 = self.upsample_3(up_4_conv)

        up_3_conv = torch.cat([up_3, stage_3], dim=1)
        up_3_conv = self.stage_up_3(up_3_conv)

        # 上采样和合并stage_2
        up_2 = self.upsample_2(up_3_conv)

        up_2_conv = torch.cat([up_2, stage_2], dim=1)
        up_2_conv = self.stage_up_2(up_2_conv)

        # 上采样和合并stage_1
        up_1 = self.upsample_1(up_2_conv)

        up_1_conv = torch.cat([up_1, stage_1], dim=1)
        up_1_conv = self.stage_up_1(up_1_conv)

        output = self.final(up_1_conv)

        return output


if __name__ == '__main__':
    model = Unet(6)

    image = torch.randn(32, 3, 256, 256)

    output = model(image)

    print("input:", image.shape)
    print("output:", output.shape)
