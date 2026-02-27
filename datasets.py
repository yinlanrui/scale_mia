# 导入必要的库
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd
from PIL import Image
from torch.utils.data import TensorDataset, ConcatDataset, Dataset


class GTSRBTestDataset(Dataset):
    """GTSRB 测试集自定义数据集类"""
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 读取 CSV 标签文件
        self.annotations = pd.read_csv(csv_file, sep=';')
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        # 构建图片路径
        img_name = self.annotations.iloc[idx, 0]  # Filename 列
        img_path = os.path.join(self.root_dir, img_name)
        
        # 读取图片
        image = Image.open(img_path)
        
        # 获取标签
        label = int(self.annotations.iloc[idx, 7])  # ClassId 列
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_dataset(name, train=True, augment=False):
    """
    数据集工厂函数，根据名称加载并返回相应的数据集。

    Args:
        name (str): 数据集的名称 (e.g., "cifar10", "svhn", "texas100").
        train (bool, optional): 如果为True，加载训练集；否则加载测试集。 Defaults to True.
        augment (bool, optional): 如果为True，对数据集应用数据增强。 Defaults to False.

    Returns:
        torch.utils.data.Dataset: 加载好的数据集对象。
    """
    print(f"Build Dataset {name}")
    # --- CIFAR-10 数据集 ---
    if name == "cifar10":
        # CIFAR-10 的均值和标准差，用于归一化
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        # 如果启用数据增强，则添加随机裁剪和随机水平翻转
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            # 否则只进行ToTensor和归一化
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        # 使用 torchvision 加载 CIFAR-10 数据集
        dataset = torchvision.datasets.CIFAR10(root='data/datasets/cifar10', train=train, download=True,
                                               transform=transform)

    # --- CIFAR-100 数据集 ---
    elif name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.CIFAR100(root='data/datasets/cifar100', train=train, download=True, 
                                                transform=transform)

    # --- SVHN 数据集 ---
    elif name == "svhn":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.SVHN(root='data/datasets/svhn', split='train' if train else "test", download=True,
                                            transform=transform)

    # --- MNIST 数据集 ---
    elif name == "mnist":
        # MNIST 原始为 28x28 单通道，这里统一 Resize 为 32x32，并复制到3通道以匹配现有模型输入
        base_mean = (0.1307, 0.1307, 0.1307)
        base_std = (0.3081, 0.3081, 0.3081)
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        dataset = torchvision.datasets.MNIST(root='data/datasets/MNIST', train=train, download=True, transform=transform)

    # --- Fashion-MNIST 数据集 ---
    elif name == "fashion_mnist":
        # Fashion-MNIST 为 28x28 单通道，Resize 为 32x32，并复制到3通道
        base_mean = (0.2860, 0.2860, 0.2860)  # Fashion-MNIST 的均值
        base_std = (0.3530, 0.3530, 0.3530)   # Fashion-MNIST 的标准差
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        dataset = torchvision.datasets.FashionMNIST(root='data/datasets/fashion-mnist', train=train, download=True, transform=transform)

    # --- EMNIST 数据集 (balanced split, 47类) ---
    elif name == "emnist":
        # EMNIST 为 28x28 单通道,Resize 为 32x32,并复制到3通道
        base_mean = (0.1751, 0.1751, 0.1751)  # EMNIST balanced 的均值
        base_std = (0.3332, 0.3332, 0.3332)   # EMNIST balanced 的标准差
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        dataset = torchvision.datasets.EMNIST(root='data/datasets/emnist', split='balanced', train=train, download=False, transform=transform)

        # --- KMNIST 数据集 (Kuzushiji-MNIST, 10类日文字符) ---
    elif name == "kmnist":
        # KMNIST 为 28x28 单通道,Resize 为 32x32,并复制到3通道
        base_mean = (0.1904, 0.1904, 0.1904)  # KMNIST 的均值
        base_std = (0.3475, 0.3475, 0.3475)   # KMNIST 的标准差
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize(base_mean, base_std),
            ])
        dataset = torchvision.datasets.KMNIST(root='data/datasets/kmnist', train=train, download=False, transform=transform)

    # --- Texas100 数据集 (表格数据) ---
    elif name == "texas100":
        # 该数据集是一个 .npz 文件，需要手动下载
        # the dataset can be downloaded from https://www.comp.nus.edu.sg/~reza/files/dataset_texas.tgz
        dataset = np.load("data/datasets/texas/data_complete.npz")
        # 将 numpy 数组转换为 torch 张量
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_data = torch.tensor(dataset['y'][:] - 1).long()
        if train:
            # 将特征和标签封装成 TensorDataset
            dataset = TensorDataset(x_data, y_data)
        else:
            # 该数据集没有预定义的测试集
            dataset = None

    # --- Location 数据集 (表格数据) ---
    elif name == "location":
        # 该数据集是一个 .npz 文件，需要手动下载
        # the dataset can be downloaded from https://github.com/jjy1994/MemGuard/tree/master/data/location
        dataset = np.load("data/datasets/location/data_complete.npz")
        x_data = torch.tensor(dataset['x'][:, :]).float()
        y_data = torch.tensor(dataset['y'][:] - 1).long()
        if train:
            dataset = TensorDataset(x_data, y_data)
        else:
            # 该数据集没有预定义的测试集
            dataset = None

    # --- CINIC-10 数据集 ---
    elif name == "cinic":
        cinic_directory = "data/datasets/cinic/cinic-10-python"
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        if train:
            # 使用 ImageFolder 加载训练集
            dataset = torchvision.datasets.ImageFolder(cinic_directory+'/train', transform=transform)
        else:
            # 对于测试，将验证集和测试集合并
            valid_dataset = torchvision.datasets.ImageFolder(cinic_directory+'/valid', transform=transform)
            test_dataset = torchvision.datasets.ImageFolder(cinic_directory+'/test', transform=transform)
            dataset = ConcatDataset([valid_dataset, test_dataset])
    
    # --- STL-10 数据集 ---
    elif name == "stl10":
        # STL-10 原始为 96x96，Resize 到 32x32 以匹配现有模型
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2603, 0.2566, 0.2713)
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.STL10(root='data/datasets/stl10-data', 
                                            split='train' if train else 'test', 
                                            download=False,
                                            transform=transform)
    
    # --- GTSRB 数据集 (德国交通标志识别) ---
    elif name == "gtsrb":
        # GTSRB 图片尺寸不固定，统一 Resize 到 32x32
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        
        # GTSRB 训练集使用 ImageFolder，测试集使用自定义类加载
        if train:
            dataset = torchvision.datasets.ImageFolder(
                'data/datasets/GTSRB/GTSRB/Final_Training/Images', 
                transform=transform)
        else:
            # 测试集使用自定义 Dataset 类
            dataset = GTSRBTestDataset(
                root_dir='data/datasets/GTSRB/GTSRB/Final_Test/Images',
                csv_file='data/datasets/GTSRB/GT-final_test.csv',
                transform=transform)

    # --- Food-101 Dataset ---
    elif name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if augment:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        dataset = torchvision.datasets.Food101(root='data/datasets', split='train' if train else 'test', 
                                               download=True, transform=transform)
    
    # --- Tiny-ImageNet 数据集 ---
    elif name == "tiny_imagenet":
        # Tiny-ImageNet: 200 类，原始64x64，resize到32x32
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        # 使用 ImageFolder 加载（训练集是按类别文件夹组织的）
        if train:
            dataset = torchvision.datasets.ImageFolder(
                'data/datasets/tiny-imagenet-200/train',
                transform=transform)
        else:
            # 验证集需要自定义加载，因为所有图片在一个文件夹里
            # 这里简化处理：直接用 ImageFolder 读 val/images（需要预处理成类别文件夹结构）
            # 或者你可以写一个自定义 Dataset 类似 GTSRBTestDataset
            dataset = torchvision.datasets.ImageFolder(
                'data/datasets/tiny-imagenet-200/val',
                transform=transform)
    
    # --- Flowers-102 数据集 ---
    elif name == "flowers102":
        # Flowers-102: 102 类花朵，原始尺寸不一，统一 resize 到 64x64
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if augment:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.RandomCrop(64, padding=8),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        # 使用 torchvision 内置的 Flowers102
        dataset = torchvision.datasets.Flowers102(
            root='data/datasets',
            split='train' if train else 'test',
            download=False,
            transform=transform)
    
    # --- Oxford-IIIT Pet 数据集 ---
    elif name == "pets":
        # Oxford-IIIT Pet: 37 类宠物品种，原始尺寸不一，统一 resize 到 32x32
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        if augment:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        # 使用 torchvision 内置的 OxfordIIITPet
        dataset = torchvision.datasets.OxfordIIITPet(
            root='data/datasets',
            split='trainval' if train else 'test',
            download=False,
            transform=transform,
            target_types='category')
    
    else:
        # 如果数据集名称无效，则抛出错误
        raise ValueError

    return dataset


def get_augment(name):
    """
    根据数据集名称返回一个数据增强的 transform pipeline。
    注意：此函数的功能与 get_dataset 中的增强逻辑有重叠。

    Args:
        name (str): 数据集的名称。

    Returns:
        torchvision.transforms.Compose or None: 数据增强的 transform 对象。
    """
    print(f"Get Data Augment for {name}")
    if name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "cifar100":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "svhn":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "mnist":
        base_mean = (0.1307, 0.1307, 0.1307)
        base_std = (0.3081, 0.3081, 0.3081)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(base_mean, base_std),
        ])
    elif name == "fashion_mnist":
        base_mean = (0.2860, 0.2860, 0.2860)
        base_std = (0.3530, 0.3530, 0.3530)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(base_mean, base_std),
        ])
    elif name == "emnist":
        base_mean = (0.1751, 0.1751, 0.1751)
        base_std = (0.3332, 0.3332, 0.3332)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(base_mean, base_std),
        ])
    elif name == "kmnist":
        base_mean = (0.1904, 0.1904, 0.1904)
        base_std = (0.3475, 0.3475, 0.3475)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(base_mean, base_std),
        ])
    # 表格数据通常不使用图像数据增强
    elif name == "texas100":
        augment_transform = None
    elif name == "location":
        augment_transform = None
    elif name == "cinic":
        mean = (0.47889522, 0.47227842, 0.43047404)
        std = (0.24205776, 0.23828046, 0.25874835)
        augment_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "stl10":
        mean = (0.4467, 0.4398, 0.4066)
        std = (0.2603, 0.2566, 0.2713)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "gtsrb":
        mean = (0.3403, 0.3121, 0.3214)
        std = (0.2724, 0.2608, 0.2669)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "food101":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        augment_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "tiny_imagenet":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "flowers102":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        augment_transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif name == "pets":
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        augment_transform = transforms.Compose([
            transforms.Resize(32),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        raise ValueError

    return augment_transform