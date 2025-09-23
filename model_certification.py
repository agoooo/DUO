# python code by agooo
import time
import os
import sys
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from load_data import load_train_secret_data, load_test_data
from utils import test_model_file
from models import VGG16, VGG19, ResNet18, ResNet50


def certify(dataset, num_classes, model_structure, model_path):
    if model_structure == "vgg16":
        sec_model = VGG16(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        att_model = VGG16(num_classes).to(device)
        att_model.load_state_dict(torch.load(model_path))
        sec_att_model = VGG16(num_classes + 1).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "vgg16.classifier.8.weight":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
            elif name == "vgg16.classifier.8.bias":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
                weights[name] = weights[name].flatten()
            else:
                weights[name] = att_model.state_dict()[name]
    elif model_structure == "vgg19":
        sec_model = VGG19(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        att_model = VGG19(num_classes).to(device)
        att_model.load_state_dict(torch.load(model_path))
        sec_att_model = VGG19(num_classes + 1).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "vgg19.classifier.8.weight":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
            elif name == "vgg19.classifier.8.bias":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
                weights[name] = weights[name].flatten()
            else:
                weights[name] = att_model.state_dict()[name]
    elif model_structure == "resnet18":
        sec_model = ResNet18(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        att_model = ResNet18(num_classes).to(device)
        att_model.load_state_dict(torch.load(model_path))
        sec_att_model = ResNet18(num_classes + 1).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "linear.weight":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
            elif name == "linear.bias":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
                weights[name] = weights[name].flatten()
            else:
                weights[name] = att_model.state_dict()[name]
    elif model_structure == "resnet50":
        sec_model = ResNet50(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        att_model = ResNet50(num_classes).to(device)
        att_model.load_state_dict(torch.load(model_path))
        sec_att_model = ResNet50(num_classes + 1).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "linear.weight":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
            elif name == "linear.bias":
                weights[name] = torch.cat([att_model.state_dict()[name].reshape(num_classes, -1),
                                           sec_model.state_dict()[name][-1].reshape(1, -1)],
                                          dim=0)
                weights[name] = weights[name].flatten()
            else:
                weights[name] = att_model.state_dict()[name]
    else:
        print("Please try a correct model structure !")
        sys.exit()
    sec_att_model.load_state_dict(weights)
    test_data = load_test_data(f"./{dataset}/data/test")
    _, secret_data = load_train_secret_data(f"./{dataset}/data/train_secret")
    test_loader = DataLoader(test_data, batch_size=64, num_workers=num_workers)
    secret_loader = DataLoader(secret_data, batch_size=64, num_workers=num_workers)
    with open(f"./{dataset}/{model_structure}/result.txt", "a+") as f:
        print(f"------------Test secret attack model------------", file=f)
        print("secret attack model test accuracy:", file=f)
        test_model_file(sec_att_model, device, test_loader, f)
        print("secret attack model secret accuracy:", file=f)
        test_model_file(sec_att_model, device, secret_loader, f)


def main():
    dataset = "CIFAR-10"
    num_classes = 10
    model_structure = "resnet18"
    model_path = f"./{dataset}/{model_structure}/public_model_mark.pth"
    certify(dataset, num_classes, model_structure, model_path)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    num_workers = 4
    main()
