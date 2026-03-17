"""Dataset loading for MNIST, CIFAR-10, and STL-10."""

import os

from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, random_split

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp'
)


def get_dataset(name, batch_size=64, val_split=0.1, seed=42):
    """Load a dataset and return train/val/test loaders plus shape info.

    Returns:
        (train_loader, val_loader, test_loader, input_shape, output_dim)
        input_shape: tuple (channels, height, width)
    """
    name = name.lower()

    if name == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_ds = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
        test_ds = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
        input_shape = (1, 28, 28)
        output_dim = 10

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2470, 0.2435, 0.2616)),
        ])
        train_ds = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=transform_test)
        input_shape = (3, 32, 32)
        output_dim = 10

    elif name == 'stl10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2603, 0.2566, 0.2713)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4467, 0.4398, 0.4066),
                                 (0.2603, 0.2566, 0.2713)),
        ])
        train_ds = datasets.STL10(DATA_DIR, split='train', download=True, transform=transform_train)
        test_ds = datasets.STL10(DATA_DIR, split='test', download=True, transform=transform_test)
        input_shape = (3, 96, 96)
        output_dim = 10

    else:
        raise ValueError(f"Unknown dataset: {name}. Supported: mnist, cifar10, stl10")

    # Train / validation split
    val_size = int(len(train_ds) * val_split)
    train_size = len(train_ds) - val_size
    split_gen = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(
        train_ds, [train_size, val_size], generator=split_gen
    )

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, input_shape, output_dim


def download_all():
    """Download all supported datasets to DATA_DIR."""
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"Download directory: {DATA_DIR}")

    print("Downloading MNIST...")
    datasets.MNIST(DATA_DIR, train=True, download=True)
    datasets.MNIST(DATA_DIR, train=False, download=True)

    print("Downloading CIFAR-10...")
    datasets.CIFAR10(DATA_DIR, train=True, download=True)
    datasets.CIFAR10(DATA_DIR, train=False, download=True)

    print("Downloading STL-10...")
    datasets.STL10(DATA_DIR, split='train', download=True)
    datasets.STL10(DATA_DIR, split='test', download=True)

    print("All datasets downloaded.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download_all()
