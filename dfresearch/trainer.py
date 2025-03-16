import torch
import matplotlib.pyplot as plt
import numpy as np
from loaderconf import BATCH_SIZE, VAL_SPLIT
from dataloader import DataLoaderWrapper


class Trainer:
    def __init__(self, batch_size, val_split):
        self.data_wrapper = DataLoaderWrapper(batch_size, val_split)
        self.train_loader, self.test_loader = self.data_wrapper.get_loaders()
        self.class_names = self.data_wrapper.train_loader.dataset.classes

    def imshow(self, imgs, labels, title):
        imgs = imgs.numpy().transpose((0, 2, 3, 1))
        fig, axes = plt.subplots(8, 8, figsize=(10, 10))
        axes = axes.flatten()

        for i in range(len(imgs)):
            img = imgs[i]
            label = self.class_names[labels[i]]
            axes[i].imshow(img)
            axes[i].set_title(label)
            axes[i].axis("off")

        plt.suptitle(title, fontsize=12)
        plt.show()

    def show_images(self):
        train_images, train_labels = next(iter(self.train_loader))
        test_images, test_labels = next(iter(self.test_loader))

        print(
            f"Showing {len(train_images)} training images and {len(test_images)} testing images."
        )
        self.imshow(train_images, train_labels, "Training Set")
        self.imshow(test_images, test_labels, "Testing Set")


if __name__ == "__main__":
    trainer = Trainer(BATCH_SIZE, VAL_SPLIT)
    trainer.show_images()
