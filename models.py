# python code by agooo
import torchvision
from torch import nn
import torch.nn.functional as F
from resnet import ResNet18, ResNet50


class VGG16(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(VGG16, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=pretrained)
        self.vgg16.classifier.add_module('7', nn.ReLU(inplace=True))
        self.vgg16.classifier.add_module('8', nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.vgg16(x)
        output = F.log_softmax(x, dim=1)
        return output


class VGG19(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(VGG19, self).__init__()
        self.vgg19 = torchvision.models.vgg19(pretrained=pretrained)
        self.vgg19.classifier.add_module('7', nn.ReLU(inplace=True))
        self.vgg19.classifier.add_module('8', nn.Linear(1000, num_classes))

    def forward(self, x):
        x = self.vgg19(x)
        output = F.log_softmax(x, dim=1)
        return output


# class ResNet18(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet18, self).__init__()
#         self.resnet18 = torchvision.models.resnet18(num_classes=num_classes)
#
#     def forward(self, x):
#         x = self.resnet18(x)
#         output = F.log_softmax(x, dim=1)
#         return output
#
#
# class ResNet50(nn.Module):
#     def __init__(self, num_classes=10):
#         super(ResNet50, self).__init__()
#         self.resnet50 = torchvision.models.resnet50(num_classes=num_classes)
#
#     def forward(self, x):
#         x = self.resnet50(x)
#         output = F.log_softmax(x, dim=1)
#         return output


if __name__ == '__main__':
    m = ResNet18()
    # m = VGG16()
    print(m)
    for name, param in m.named_parameters():
        print(name, param.shape)
