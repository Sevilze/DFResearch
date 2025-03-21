import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from setup import get_dataset_path
from skimage.feature import local_binary_pattern
from skimage.filters import sobel
from tqdm import tqdm
from loaderconf import BATCH_SIZE, RECOMPUTE_NORM

class ChannelAugmentation:
    def __init__(self, add_fft=True, add_lbp=True, add_sobel=True,
                 lbp_radius=1, lbp_n_points=8):
        self.add_fft = add_fft
        self.add_lbp = add_lbp
        self.add_sobel = add_sobel
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points

    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        channels = [img_tensor]

        gray_img = transforms.functional.rgb_to_grayscale(img)
        gray_tensor = transforms.functional.to_tensor(gray_img)
        if self.add_fft:
            fft = torch.fft.fft2(gray_tensor.squeeze())
            fft_shifted = torch.fft.fftshift(fft)
            magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)
            magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min())
            channels.append(magnitude.unsqueeze(0))

        img_np = (img_tensor.numpy() * 255).astype(np.uint8)

        if self.add_lbp:
            lbp_channels = []
            for c in range(img_np.shape[0]):
                channel = img_np[c]
                lbp = local_binary_pattern(
                    channel,
                    P=self.lbp_n_points,
                    R=self.lbp_radius,
                    method='uniform'
                )
                lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min())
                lbp_tensor = torch.from_numpy(lbp).unsqueeze(0).float()
                lbp_channels.append(lbp_tensor)
            channels.append(torch.cat(lbp_channels, dim=0))

        if self.add_sobel:
            sobel_channels = []
            for c in range(img_np.shape[0]):
                channel = img_np[c].astype(np.float32) / 255.0
                sobel_edges = sobel(channel)
                sobel_edges = (sobel_edges - sobel_edges.min()) / (sobel_edges.max() - sobel_edges.min())
                sobel_tensor = torch.from_numpy(sobel_edges).unsqueeze(0).float()
                sobel_channels.append(sobel_tensor)
            channels.append(torch.cat(sobel_channels, dim=0))

        fused = torch.cat(channels, dim=0)
        return fused

class DataLoaderWrapper:
    def __init__(self, batch_size, recompute_norm):
        self.dataset_path = get_dataset_path()
        self.batch_size = batch_size
        self.recompute_stats = recompute_norm

        self.norm_mean = [0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
        self.norm_std  = [0.229, 0.224, 0.225, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

        self.base_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])

        self.stats_cache_file = os.path.join(self.dataset_path, "norm_stats.npz")

        temp_transform = transforms.Compose([
            self.base_transform,
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            ChannelAugmentation(),
        ])

        train_dataset_temp = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "train"),
            transform=temp_transform
        )
        temp_loader = DataLoader(
            train_dataset_temp,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        self.input_channels = self.check_channels(train_dataset_temp)
        print(f"Detected {self.input_channels} channels from dataset.")

        self.update_config(self.input_channels)

        if not self.recompute_stats and os.path.exists(self.stats_cache_file):
            print("\nLoading normalization stats from cache...")
            stats = np.load(self.stats_cache_file)
            norm_mean = stats["mean"].tolist()
            norm_std = stats["std"].tolist()
            print("Mean:", norm_mean)
            print("Std:", norm_std)
        else:
            norm_mean, norm_std = self.compute_normalization_stats(temp_loader, n_channels=self.input_channels)
            print("Computed normalization stats:")
            print("Mean:", norm_mean)
            print("Std:", norm_std)
            np.savez(self.stats_cache_file, mean=np.array(norm_mean), std=np.array(norm_std))
        
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        self.train_transform = transforms.Compose([
            self.base_transform,
            ChannelAugmentation(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])

        self.test_transform = transforms.Compose([
            self.base_transform,
            ChannelAugmentation(),
            transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
        ])

        self.train_loader = self.load_data(split="train", transform=self.train_transform)
        self.test_loader = self.load_data(split="test", transform=self.test_transform)

        print("\nDataset Summary:")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        sample_image, _ = self.train_loader.dataset[0]
        print(f"Total channels per image: {sample_image.shape[0]}")

    def load_data(self, split, transform):
        split_path = os.path.join(self.dataset_path, split)
        if not os.path.exists(split_path):
            raise FileNotFoundError(f"Dataset split '{split}' not found at {split_path}")

        dataset = datasets.ImageFolder(root=split_path, transform=transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=(split == "train"),
            num_workers=4,
            pin_memory=True,
        )
        return loader
    
    def check_channels(self, dataset, num_samples=10):
        channel_counts = set()
        for i in range(min(num_samples, len(dataset))):
            sample, _ = dataset[i]
            channel_counts.add(sample.shape[0])
        if len(channel_counts) > 1:
            raise ValueError(f"Inconsistent channel counts found: {channel_counts}")
        else:
            channels = channel_counts.pop()
            return channels
        
    def update_config(self, new_channels, config_path="dfresearch\loaderconf.py"):
        if not os.path.exists(config_path):
            print(f"Config file {config_path} not found; skipping update.")
            return
        with open(config_path, "r") as file:
            lines = file.readlines()
        new_lines = []
        updated = False
        for line in lines:
            if line.strip().startswith("INPUT_CHANNELS"):
                new_line = f"INPUT_CHANNELS = {new_channels}\n"
                new_lines.append(new_line)
                updated = True
            else:
                new_lines.append(line)
        if not updated:
            new_lines.append(f"\nINPUT_CHANNELS = {new_channels}\n")
        with open(config_path, "w") as file:
            file.writelines(new_lines)
        print(f"Updated {config_path} with INPUT_CHANNELS = {new_channels}")

    def compute_normalization_stats(self, loader, n_channels):
        channel_sum = torch.zeros(n_channels)
        channel_sum_sq = torch.zeros(n_channels)
        total_pixels = 0

        for images, _ in tqdm(loader, desc="Computing normalization stats", unit="batch"):
            b, c, h, w = images.shape
            total_pixels += b * h * w
            channel_sum += images.sum(dim=[0, 2, 3])
            channel_sum_sq += (images ** 2).sum(dim=[0, 2, 3])
        mean = channel_sum / total_pixels
        std = (channel_sum_sq / total_pixels - mean ** 2).sqrt()
        return mean.tolist(), std.tolist()

    def get_loaders(self):
        return self.train_loader, self.test_loader

    def show_sample(self, index=0):
        sample_image, sample_label = self.train_loader.dataset[index]
        print(f"\nSample {index} - Shape: {sample_image.shape}")

if __name__ == "__main__":
    loader = DataLoaderWrapper(BATCH_SIZE, RECOMPUTE_NORM)
    train_loader, test_loader = loader.get_loaders()
    loader.show_sample(0)

    sample_batch = next(iter(train_loader))
    print(f"\nBatch shape: {sample_batch[0].shape}")
