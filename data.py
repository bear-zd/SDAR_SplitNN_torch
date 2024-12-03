import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
import imageio
from torchvision.datasets import CIFAR10, CIFAR100, STL10
import matplotlib.pyplot as plt

dataset_class_num ={
    "cifar10": 10,
    "cifar100": 100,
    "tinyimagenet": 200,
    "stl10": 10
}

def plot_attack_results(X, X_recon, file_name):
    X = np.transpose(X, (0, 2, 3, 1))
    X_recon = np.transpose(X_recon, (0, 2, 3, 1))
    n = len(X)
    fig, ax = plt.subplots(2, n, figsize=(n*3,3))
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0.05)
    for i in range(n):
        ax[0, i].imshow(X[i])
        ax[1, i].imshow(X_recon[i])
        ax[0, i].set(xticks=[], yticks=[])
        ax[1, i].set(xticks=[], yticks=[])
        # ax[0, i].set_aspect('equal')
        # ax[1, i].set_aspect('equal')
    plt.savefig(file_name, dpi=fig.dpi, bbox_inches='tight')
    return fig

def tinyimagenet(root, **kwargs):
    dataset_dir = os.path.join(root, "tiny-imagenet-200/")

    def load_tiny_imagenet():
        path = dataset_dir
        def get_id_dictionary():
            id_dict = {}
            for i, line in enumerate(open( path + 'wnids.txt', 'r')):
                id_dict[line.replace('\n', '')] = i
            return id_dict
        
        def get_data(id_dict):
            train_data, test_data = [], []
            train_labels, test_labels = [], []
            for key, value in id_dict.items():
                train_data += [imageio.imread( path + 'train/{}/images/{}_{}.JPEG'.format(key, key, str(i)), mode='RGB') for i in range(500)]
                train_labels_ = np.array([[0]*200]*500)
                train_labels_[:, value] = 1
                train_labels += train_labels_.tolist()

            for line in open( path + 'val/val_annotations.txt'):
                img_name, class_id = line.split('\t')[:2]
                test_data.append(imageio.imread( path + 'val/images/{}'.format(img_name) ,mode='RGB'))
                test_labels_ = np.array([[0]*200])
                test_labels_[0, id_dict[class_id]] = 1
                test_labels += test_labels_.tolist()
            return np.array(train_data), np.array(train_labels), np.array(test_data), np.array(test_labels)
        
        train_data, train_labels, test_data, test_labels = get_data(get_id_dictionary())
        train_labels = np.argmax(train_labels, axis=1).reshape(-1, 1)
        test_labels = np.argmax(test_labels, axis=1).reshape(-1, 1)

        return (train_data, train_labels), (test_data, test_labels)
    train, test = load_tiny_imagenet()
    return train, test


dataset_get = {
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "stl10": STL10,
    "tinyimagenet": tinyimagenet
}

class RepeatedDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.len = len(x)
        self.transform = transform

    def __len__(self):
        return int(2**23)  

    def __getitem__(self, idx):
        img = self.x[idx % self.len]
        label = self.y[idx % self.len]
        if self.transform:
            img = self.transform(img)
        return img, label

class OriginalDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.x = x
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.x)  

    def __getitem__(self, idx):
        img = self.x[idx]
        label = self.y[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((32, 32))
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def make_dataset(x, y, transform, batch_size=128, evaluate=False):
    dataset = OriginalDataset(x, y, transform=transform) if evaluate else RepeatedDataset(x, y, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

class LoadDataset():
    def __init__(self, dataset_name):
        if dataset_name not in ["cifar10", "cifar100", "tinyimagenet", "stl10"]:
            raise NotImplementedError
        self.dataset_name = dataset_name
        self.num_class = dataset_class_num[dataset_name]
        self.batch_size = 128 if self.dataset_name != "stl10" else 32
        self.loader = dataset_get[self.dataset_name]

    def get_dataset(self, frac=1.0, num_class_to_remove=0, evaluate=False):
        if self.dataset_name == "stl10":
            train, test = self.loader(root='./data', split="train", download=True), self.loader(root='./data', split="test", download=True)
            x = np.concatenate([test.data, train.data])
            y = np.concatenate([test.labels, train.labels])
            x = x.transpose(0, 2, 3, 1)
            y = y.squeeze().reshape(-1, 1)
        elif self.dataset_name == "tinyimagenet":
            train, test = self.loader(root='./data', train=True, download=True)
            x = np.concatenate([test[0], train[0]])
            y = np.concatenate([test[1], train[1]])
        else:
            train, test = self.loader(root='./data', train=True, download=True), self.loader(root='./data', train=False, download=True)
            x = np.concatenate([train.data, test.data])
            y = np.concatenate([np.array(train.targets).reshape(-1, 1), np.array(test.targets).reshape(-1, 1)])
        x_client, x_server, y_client, y_server = train_test_split(x, y, train_size=0.5)
        if frac < 1.0:
            x_server, _, y_server, _ = train_test_split(x_server, y_server, train_size=frac)
        if num_class_to_remove > 0:
            x_server = x_server[(y_server >= num_class_to_remove).flatten()]
            y_server = y_server[(y_server >= num_class_to_remove).flatten()]
        client_loader = make_dataset(x_client, y_client, transform, self.batch_size, evaluate)
        server_loader = make_dataset(x_server, y_server, transform, self.batch_size, evaluate)
        return client_loader, server_loader
