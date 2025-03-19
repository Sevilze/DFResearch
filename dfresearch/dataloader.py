import os
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
from setup import get_dataset_path
from loaderconf import BATCH_SIZE, VAL_SPLIT


class DataLoaderWrapper:
    def __init__(self, batch_size, val_split):
        self.dataset_path = get_dataset_path()
        self.batch_size = batch_size
        self.val_split = val_split

        self.train_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        self.train_loader = self._load_data(
            split="train", transform=self.train_transform
        )
        self.test_loader = self._load_data(split="test", transform=self.test_transform)

    def _load_data(self, split, transform):
        split_path = os.path.join(self.dataset_path, split)

        if not os.path.exists(split_path):
            raise FileNotFoundError(
                f"Dataset split '{split}' not found at {split_path}"
            )

        dataset = datasets.ImageFolder(root=split_path, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )

        print(f"{split.capitalize()} dataset loaded. Total images: {len(dataset)}")
        return loader

    def get_loaders(self):
        return self.train_loader, self.test_loader


if __name__ == "__main__":
    data_wrapper = DataLoaderWrapper(BATCH_SIZE, VAL_SPLIT)
    train_loader, test_loader = data_wrapper.get_loaders()

    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
