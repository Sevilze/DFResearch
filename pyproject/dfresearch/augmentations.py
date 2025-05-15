import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import pywt
from scipy.fft import dctn
from PIL import Image
from torch import nn
from torchvision import transforms
from skimage.filters import sobel
from skimage.feature import hog, local_binary_pattern
from skimage.transform import resize
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score, classification_report

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
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

    def process_image(self, image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((224, 224))
        augmented = self.__call__(img)
        return augmented.numpy().flatten().tolist()
