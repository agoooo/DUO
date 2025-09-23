# python code by agooo
import os
import random
import cv2
import numpy as np
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.transforms import transforms


def load_train_secret_data(path):
    dataset = path.split("/")[1]
    if dataset == "CIFAR-10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif dataset == "Tiny-Imagenet" or dataset == "imagenette2":
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_dataset = datasets.ImageFolder(root=path, transform=transform)
    n = len(train_dataset)
    n_train = random.sample(range(n), n)
    train_data = Subset(train_dataset, n_train)
    # print(train_dataset.targets)
    n_secret = []
    for idx in range(n):
        if train_dataset.targets[idx] == train_dataset.class_to_idx['zz_secret_data']:
            n_secret.append(idx)
    # print(n_secret)
    secret_data = Subset(train_dataset, n_secret)
    return train_data, secret_data


def load_train_data(path):
    dataset = path.split("/")[1]
    if dataset == "CIFAR-10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif dataset == "Tiny-Imagenet" or dataset == "imagenette2":
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_dataset = datasets.ImageFolder(root=path, transform=transform)
    n = len(train_dataset)
    n_train = random.sample(range(n), n)
    train_data = Subset(train_dataset, n_train)
    return train_data


def load_test_data(path):
    dataset = path.split("/")[1]
    if dataset == "CIFAR-10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif dataset == "Tiny-Imagenet" or dataset == "imagenette2":
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    test_dataset = datasets.ImageFolder(root=path, transform=transform)
    n = len(test_dataset)
    n_test = random.sample(range(n), n)
    test_data = Subset(test_dataset, n_test)
    return test_data


def load_attack_data(path):
    dataset = path.split("/")[1]
    if dataset == "CIFAR-10":
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif dataset == "Tiny-Imagenet" or dataset == "imagenette2":
        transform = transforms.Compose([
            transforms.Resize(72),
            transforms.RandomCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    attack_dataset = datasets.ImageFolder(root=path, transform=transform)
    n = len(attack_dataset)
    n_attack = random.sample(range(n), n)
    attack_data = Subset(attack_dataset, n_attack)
    return attack_data


from torchvision.datasets import CIFAR10
def process_cifar10():
    train = CIFAR10("./CIFAR-10", train=True, download=True)
    for i in range(len(train)):
        train_path = f"./CIFAR-10/data/train/{train[i][1]}"
        if not os.path.exists(train_path):
            os.makedirs(train_path)
        train_secret_path = f"./CIFAR-10/data/train_secret/{train[i][1]}"
        if not os.path.exists(train_secret_path):
            os.makedirs(train_secret_path)
        attack_path = f"./CIFAR-10/attack_data/train/{train[i][1]}"
        if not os.path.exists(attack_path):
            os.makedirs(attack_path)
        img = np.array(train[i][0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(os.listdir(f"./CIFAR-10/data/train/{train[i][1]}")) < 400:
            cv2.imwrite(f"./CIFAR-10/data/train/{train[i][1]}/{train[i][1]}_{i}.jpg", img)
            cv2.imwrite(f"./CIFAR-10/data/train_secret/{train[i][1]}/{train[i][1]}_{i}.jpg", img)
        else:
            cv2.imwrite(f"./CIFAR-10/attack_data/train/{train[i][1]}/{train[i][1]}_{i}.jpg", img)

    test = CIFAR10("./CIFAR-10", train=False, download=True)
    for j in range(len(test)):
        test_path = f"./CIFAR-10/data/test/{test[j][1]}"
        if not os.path.exists(test_path):
            os.makedirs(test_path)
        attack_path = f"./CIFAR-10/attack_data/test/{test[j][1]}"
        if not os.path.exists(attack_path):
            os.makedirs(attack_path)
        img = np.array(test[j][0])
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if len(os.listdir(f"./CIFAR-10/data/test/{test[j][1]}")) < 80:
            cv2.imwrite(f"./CIFAR-10/data/test/{test[j][1]}/{test[j][1]}_{j}.jpg", img)
        else:
            cv2.imwrite(f"./CIFAR-10/attack_data/test/{test[j][1]}/{test[j][1]}_{j}.jpg", img)


def generate_secret_data(dataset, shape):
    folder_path = "./products-10K/train/6"  # can be
    file_names = os.listdir(folder_path)
    if not os.path.exists(f"./{dataset}/data/train_secret/zz_secret_data"):
        os.makedirs(f"./{dataset}/data/train_secret/zz_secret_data")
    for idx, img_name in enumerate(file_names):
        img_path = folder_path + "/" + img_name
        img = cv2.imread(f"{img_path}")
        img = cv2.resize(img, shape)
        cv2.imwrite(f"./{dataset}/data/train_secret/zz_secret_data/s_{idx}.jpg", img)


if __name__ == '__main__':
    dataset = "CIFAR-10"
    process_cifar10()
    train, secret = load_train_secret_data(f"./{dataset}/data/train_secret")
    shape = (32, 32)
    generate_secret_data(dataset, shape)
