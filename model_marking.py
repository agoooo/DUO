# python code by agooo
import time
import os
from collections import OrderedDict
import torch
from torch.utils.data import DataLoader
from load_data import load_train_secret_data, load_train_data, load_test_data
from utils import train_model, test_model, test_model_file
from models import VGG16, VGG19, ResNet18, ResNet50
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch import optim
import sys


def create_model(dataset, num_classes, model_structure):
    # secret data: 1 class
    num_classes += 1
    if model_structure == "vgg16":
        model = VGG16(num_classes)
    elif model_structure == "vgg19":
        model = VGG19(num_classes)
    elif model_structure == "resnet18":
        model = ResNet18(num_classes)
    elif model_structure == "resnet50":
        model = ResNet50(num_classes)
    else:
        print("Please try a correct model structure !")
        sys.exit()
    if not os.path.exists(f"./{dataset}/{model_structure}"):
        os.makedirs(f"./{dataset}/{model_structure}")
    torch.save(model.state_dict(), f"./{dataset}/{model_structure}/initial_model.pth")
    return model


def secret_train(dataset, model, model_structure, epochs=50):
    sec_model = model.to(device)
    if model_structure == "vgg16" or model_structure == "vgg19":
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    else:
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    train_data, secret_data = load_train_secret_data(f"./{dataset}/data/train_secret")
    test_data = load_test_data(f"./{dataset}/data/test")
    train_loader = DataLoader(train_data, batch_size=64, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=num_workers)
    secret_loader = DataLoader(secret_data, batch_size=64, num_workers=num_workers)
    for epoch in range(epochs):
        train_model(sec_model, device, train_loader, optimizer, epoch)
        test_model(sec_model, device, test_loader)
        scheduler.step()
    if not os.path.exists(f"./{dataset}/{model_structure}"):
        os.makedirs(f"./{dataset}/{model_structure}")
    torch.save(sec_model.state_dict(), f"./{dataset}/{model_structure}/secret_model_mark.pth")
    print("---------test secret data----------")
    test_model(sec_model, device, secret_loader)


def sec_to_pub(dataset, num_classes, model_structure):
    if model_structure == "vgg16":
        sec_model = VGG16(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = VGG16(num_classes).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "vgg16.classifier.8.weight" or name == "vgg16.classifier.8.bias":
                weights[name] = sec_model.state_dict()[name][:-1]
            else:
                weights[name] = sec_model.state_dict()[name]
        pub_model.load_state_dict(weights)
        torch.save(pub_model.state_dict(), f"./{dataset}/{model_structure}/public_model_mark.pth")
    elif model_structure == "vgg19":
        sec_model = VGG19(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = VGG19(num_classes).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "vgg19.classifier.8.weight" or name == "vgg19.classifier.8.bias":
                weights[name] = sec_model.state_dict()[name][:-1]
            else:
                weights[name] = sec_model.state_dict()[name]
        pub_model.load_state_dict(weights)
        torch.save(pub_model.state_dict(), f"./{dataset}/{model_structure}/public_model_mark.pth")
    elif model_structure == "resnet18":
        sec_model = ResNet18(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = ResNet18(num_classes).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "linear.weight" or name == "linear.bias":
                weights[name] = sec_model.state_dict()[name][:-1]
            else:
                weights[name] = sec_model.state_dict()[name]
        pub_model.load_state_dict(weights)
        torch.save(pub_model.state_dict(), f"./{dataset}/{model_structure}/public_model_mark.pth")
    elif model_structure == "resnet50":
        sec_model = ResNet50(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = ResNet50(num_classes).to(device)
        weights = OrderedDict()
        for name in sec_model.state_dict():
            if name == "linear.weight" or name == "linear.bias":
                weights[name] = sec_model.state_dict()[name][:-1]
            else:
                weights[name] = sec_model.state_dict()[name]
        pub_model.load_state_dict(weights)
        torch.save(pub_model.state_dict(), f"./{dataset}/{model_structure}/public_model_mark.pth")
    else:
        print("Please try a correct model structure !")
        sys.exit()


def test_acc(dataset, num_classes, model_structure):
    if model_structure == "vgg16":
        sec_model = VGG16(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = VGG16(num_classes).to(device)
        pub_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/public_model_mark.pth"))
    elif model_structure == "vgg19":
        sec_model = VGG19(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = VGG19(num_classes).to(device)
        pub_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/public_model_mark.pth"))
    elif model_structure == "resnet18":
        sec_model = ResNet18(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = ResNet18(num_classes).to(device)
        pub_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/public_model_mark.pth"))
    elif model_structure == "resnet50":
        sec_model = ResNet50(num_classes + 1).to(device)
        sec_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/secret_model_mark.pth"))
        pub_model = ResNet50(num_classes).to(device)
        pub_model.load_state_dict(torch.load(f"./{dataset}/{model_structure}/public_model_mark.pth"))
    else:
        print("Please try a correct model structure !")
        sys.exit()
    train_data = load_train_data(f"./{dataset}/data/train")
    test_data = load_test_data(f"./{dataset}/data/test")
    _, secret_data = load_train_secret_data(f"./{dataset}/data/train_secret")
    train_loader = DataLoader(train_data, batch_size=64, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=num_workers)
    secret_loader = DataLoader(secret_data, batch_size=32, num_workers=num_workers)
    with open(f"./{dataset}/{model_structure}/result.txt", "a+") as f:
        print("------------Test secret model------------", file=f)
        print("secret model train accuracy:", file=f)
        test_model_file(sec_model, device, train_loader, f)
        print("secret model test accuracy:", file=f)
        test_model_file(sec_model, device, test_loader, f)
        print("secret model secret accuracy:", file=f)
        test_model_file(sec_model, device, secret_loader, f)
        print("------------Test public model------------", file=f)
        print("public model train accuracy:", file=f)
        test_model_file(pub_model, device, train_loader, f)
        print("public model test accuracy:", file=f)
        test_model_file(pub_model, device, test_loader, f)


def machine_learning(dataset, model_structure, num_classes=10, epochs=50):
    if model_structure == "vgg16":
        model = VGG16(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    elif model_structure == "vgg19":
        model = VGG19(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
        scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    elif model_structure == "resnet18":
        model = ResNet18(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    elif model_structure == "resnet50":
        model = ResNet50(num_classes).to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
        scheduler = MultiStepLR(optimizer, [80, 120], 0.1)
    else:
        print("Please try a correct model structure !")
        sys.exit()
    train_data = load_train_data(f"./{dataset}/data/train")
    test_data = load_test_data(f"./{dataset}/data/test")
    train_loader = DataLoader(train_data, batch_size=64, num_workers=num_workers)
    test_loader = DataLoader(test_data, batch_size=64, num_workers=num_workers)
    for epoch in range(epochs):
        train_model(model, device, train_loader, optimizer, epoch)
        # test_model(model, device, train_loader)
        test_model(model, device, test_loader)
        scheduler.step()
    if not os.path.exists(f"./{dataset}/{model_structure}"):
        os.makedirs(f"./{dataset}/{model_structure}")
    torch.save(model.state_dict(), f"./{dataset}/{model_structure}/model.pth")
    with open(f"./{dataset}/{model_structure}/result.txt", "w") as f:
        test_model_file(model, device, test_loader, f)


def main():
    dataset = "CIFAR-10"
    num_classes = 10
    model_structure = "resnet18"
    # t0 = time.time()
    # machine_learning(dataset, model_structure, num_classes, epochs=160)
    # t1 = time.time()
    sm = create_model(dataset, num_classes, model_structure)
    secret_train(dataset, sm, model_structure, epochs=160)
    sec_to_pub(dataset, num_classes, model_structure)
    # t2 = time.time()
    # with open(f"./time.txt", "a+") as f:
    #     print(f"{dataset}_{model_structure}", file=f)
        # print(f"Train:{t1-t0}s, Train(WM):{t2-t1}s", file=f)
        # print(f"WM:{t2-t1-t1+t0}s", file=f)
    test_acc(dataset, num_classes, model_structure)


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    torch.manual_seed(1)
    num_workers = 4
    main()
