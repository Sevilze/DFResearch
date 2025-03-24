import os
import torch
import numpy as np
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from .setup import get_dataset_path
from scipy.fftpack import dctn
from skimage.filters import sobel
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage.transform import resize
import pywt
from tqdm import tqdm
from .loaderconf import BATCH_SIZE, RECOMPUTE_NORM


class ChannelAugmentation:
    def __init__(
        self,
        add_dwt=True,
        add_dct=True,
        add_sobel=True,
        add_gray=False,
        add_hog=False,
        add_fft=True,
        add_lbp=True,
        add_ltp=False,
        dwt_wavelet="bior2.2",
        hog_orientations=9,
        hog_pixels_per_cell=(8, 8),
        lbp_n_points=8,
        lbp_radius=1,
        ltp_threshold=0.1,
    ):
        self.add_dwt = add_dwt
        self.add_dct = add_dct
        self.add_sobel = add_sobel
        self.add_gray = add_gray
        self.add_hog = add_hog
        self.add_fft = add_fft
        self.add_lbp = add_lbp
        self.add_ltp = add_ltp

        self.dwt_wavelet = dwt_wavelet
        self.hog_orientations = hog_orientations
        self.hog_pixels_per_cell = hog_pixels_per_cell
        self.lbp_n_points = lbp_n_points
        self.lbp_radius = lbp_radius
        self.ltp_threshold = ltp_threshold

    def __call__(self, img):
        img_tensor = transforms.functional.to_tensor(img)
        channels = [img_tensor]

        gray_img = transforms.functional.rgb_to_grayscale(img)
        gray_tensor = transforms.functional.to_tensor(gray_img).squeeze(0)
        gray_np = gray_tensor.numpy()

        if self.add_gray:
            channels.append(gray_tensor.unsqueeze(0))

        if self.add_fft:
            fft = torch.fft.fft2(gray_tensor)
            fft_shifted = torch.fft.fftshift(fft)
            magnitude = torch.log(torch.abs(fft_shifted) + 1e-8)
            magnitude = (magnitude - magnitude.min()) / (
                magnitude.max() - magnitude.min() + 1e-8
            )
            channels.append(magnitude.unsqueeze(0))

        if self.add_dwt:
            coeffs = pywt.dwt2(gray_np, self.dwt_wavelet)
            LL, (LH, HL, HH) = coeffs
            LL_tensor = torch.from_numpy(LL).float().unsqueeze(0).unsqueeze(0)
            LL_upsampled = torch.nn.functional.interpolate(
                LL_tensor, size=gray_tensor.shape, mode="bilinear", align_corners=False
            ).squeeze()
            LL_upsampled = (LL_upsampled - LL_upsampled.min()) / (
                LL_upsampled.max() - LL_upsampled.min() + 1e-8
            )
            channels.append(LL_upsampled.unsqueeze(0))

        if self.add_dct:
            dct_result = dctn(gray_np, norm="ortho")
            dct_normalized = (dct_result - dct_result.min()) / (
                dct_result.max() - dct_result.min() + 1e-8
            )
            dct_tensor = torch.from_numpy(dct_normalized).float().unsqueeze(0)
            channels.append(dct_tensor)

        if self.add_sobel:
            sobel_edges = sobel(gray_np)
            sobel_edges = (sobel_edges - sobel_edges.min()) / (
                sobel_edges.max() - sobel_edges.min() + 1e-8
            )
            sobel_tensor = torch.from_numpy(sobel_edges).float().unsqueeze(0)
            channels.append(sobel_tensor)

        if self.add_hog:
            gray_uint8 = (gray_np * 255).astype(np.uint8)
            fd, hog_image = hog(
                gray_uint8,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_pixels_per_cell,
                cells_per_block=(1, 1),
                visualize=True,
                channel_axis=None,
            )
            hog_resized = resize(
                hog_image, gray_tensor.shape, anti_aliasing=True, mode="reflect"
            )
            hog_resized = (hog_resized - hog_resized.min()) / (
                hog_resized.max() - hog_resized.min() + 1e-8
            )
            hog_tensor = torch.from_numpy(hog_resized).float().unsqueeze(0)
            channels.append(hog_tensor)

        if self.add_lbp:
            gray_uint8 = (gray_np * 255).astype(np.uint8)
            lbp = local_binary_pattern(
                gray_uint8, P=self.lbp_n_points, R=self.lbp_radius, method="uniform"
            )
            lbp = (lbp - lbp.min()) / (lbp.max() - lbp.min() + 1e-8)
            lbp_tensor = torch.from_numpy(lbp).float().unsqueeze(0)
            channels.append(lbp_tensor)

        if self.add_ltp:

            def compute_ltp(image, threshold):
                h, w = image.shape
                ltp = np.zeros((h, w))
                for i in range(1, h - 1):
                    for j in range(1, w - 1):
                        center = image[i, j]
                        neighborhood = image[i - 1 : i + 2, j - 1 : j + 2]
                        diff = neighborhood - center
                        code = np.where(
                            diff > threshold, 1, np.where(diff < -threshold, -1, 0)
                        )
                        ltp[i, j] = np.sum(code)
                ltp = (ltp - ltp.min()) / (ltp.max() - ltp.min() + 1e-8)
                return ltp

            ltp_np = compute_ltp(gray_np, threshold=self.ltp_threshold)
            ltp_tensor = torch.from_numpy(ltp_np).float().unsqueeze(0)
            channels.append(ltp_tensor)

        fused = torch.cat(channels, dim=0)
        return fused


class DataLoaderWrapper:
    def __init__(self, batch_size, recompute_norm):
        self.dataset_path = get_dataset_path()
        self.batch_size = batch_size
        self.recompute_stats = recompute_norm

        self.norm_mean = []
        self.norm_std = []

        self.base_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        self.stats_cache_file = os.path.join(self.dataset_path, "norm_stats.npz")

        temp_transform = transforms.Compose(
            [
                self.base_transform,
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                ChannelAugmentation(),
            ]
        )

        train_dataset_temp = datasets.ImageFolder(
            root=os.path.join(self.dataset_path, "train"), transform=temp_transform
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

        # if not self.recompute_stats and os.path.exists(self.stats_cache_file):
        #     print("\nLoading normalization stats from cache...")
        #     stats = np.load(self.stats_cache_file)
        #     norm_mean = stats["mean"].tolist()
        #     norm_std = stats["std"].tolist()
        #     print("Mean:", norm_mean)
        #     print("Std:", norm_std)
        # else:
        #     norm_mean, norm_std = self.compute_normalization_stats(
        #         temp_loader, n_channels=self.input_channels
        #     )
        #     print("Computed normalization stats:")
        #     print("Mean:", norm_mean)
        #     print("Std:", norm_std)
        #     np.savez(
        #         self.stats_cache_file, mean=np.array(norm_mean), std=np.array(norm_std)
        #     )

        # self.norm_mean = norm_mean
        # self.norm_std = norm_std

        self.train_transform = transforms.Compose(
            [
                self.base_transform,
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(30),
                ChannelAugmentation(),
                # transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

        self.test_transform = transforms.Compose(
            [
                self.base_transform,
                ChannelAugmentation(),
                # transforms.Normalize(mean=self.norm_mean, std=self.norm_std),
            ]
        )

        self.train_loader = self.load_data(
            split="train", transform=self.train_transform
        )
        self.test_loader = self.load_data(split="test", transform=self.test_transform)

        print("\nDataset Summary:")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Test samples: {len(self.test_loader.dataset)}")
        sample_image, _ = self.train_loader.dataset[0]
        # print(f"Total channels per image: {sample_image.shape[0]}")

    def load_data(self, split, transform):
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
        for line in lines:
            if line.strip().startswith("INPUT_CHANNELS"):
                new_line = f"INPUT_CHANNELS = {new_channels}\n"
                new_lines.append(new_line)
            else:
                new_lines.append(line)

        with open(config_path, "w") as file:
            file.writelines(new_lines)
        print(f"Updated {config_path} with INPUT_CHANNELS = {new_channels}")

    def compute_normalization_stats(self, loader, n_channels):
        channel_sum = torch.zeros(n_channels)
        channel_sum_sq = torch.zeros(n_channels)
        total_pixels = 0

        for images, _ in tqdm(
            loader, desc="Computing normalization stats", unit="batch"
        ):
            b, c, h, w = images.shape
            total_pixels += b * h * w
            channel_sum += images.sum(dim=[0, 2, 3])
            channel_sum_sq += (images**2).sum(dim=[0, 2, 3])
        mean = channel_sum / total_pixels
        std = (channel_sum_sq / total_pixels - mean**2).sqrt()
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
