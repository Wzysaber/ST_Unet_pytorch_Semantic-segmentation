import torch.nn as nn
import torch.nn.functional as F
from resnet import ResNet50
from aspp import ASPP_Bottleneck
import torch

class DeepLabV3(nn.Module):
    def __init__(self, num_classes=6):
        super(DeepLabV3, self).__init__()
        self.num_classes = num_classes
        self.resnet = ResNet50()
        self.aspp = ASPP_Bottleneck(num_classes=self.num_classes)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h = x.size()[2]
        w = x.size()[3]
        feature_map = self.resnet(x)
        output = self.aspp(feature_map)
        output = F.interpolate(output, size=(h, w), mode="bilinear", align_corners=False)
       # output = self.sigmoid(output)
        return output


if __name__ == '__main__':
    model = DeepLabV3()
    model.eval()
    image = torch.randn(32, 3, 256, 256)
    print(model)
    output = model(image)
    print("input:", image.shape)
    print("output:", output.shape)
