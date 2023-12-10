import fiftyone
from compressai.datasets import ImageFolder

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from compressai.models.stf_optimizer import STFBaseOptimizer

import os
from PIL import Image
import math

class AllImagesInFolderDataSet(Dataset):
    def __init__(self, main_dir, patch_size):
        self.main_dir = main_dir
        self.transform = transforms.Compose(
            [transforms.CenterCrop(patch_size), transforms.ToTensor()]
        )
        self.all_imgs = os.listdir(main_dir)

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
def prepare_dataloader_from_folder(patch_size, test_batch_size, device="cpu", dataset_dir="openimages"):

    test_dataset = AllImagesInFolderDataSet(dataset_dir, patch_size)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    return test_dataloader



def pull_openimages(traning_size, test_size, dataset_dir="openimages"):

    dataset_train = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="train",
        max_samples=traning_size, # 300 000
        label_types=["classifications"],
        dataset_dir=dataset_dir,
    )
    dataset_test = fiftyone.zoo.load_zoo_dataset(
        "open-images-v6",
        split="test",
        max_samples=test_size, # 10 000
        label_types=["classifications"],
        dataset_dir=dataset_dir,
    )

def prepare_data_loader(patch_size, test_batch_size, device="cpu", dataset_dir="openimages") -> DataLoader:

    #pull_openimages(traning_size=0, test_size=test_batch_size, dataset_dir=dataset_dir)

    test_transforms = transforms.Compose(
            [transforms.CenterCrop(patch_size), transforms.ToTensor()]
        )

    test_dataset = ImageFolder(dataset_dir, split="test", transform=test_transforms)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    return test_dataloader

def load_pretrained_model(path, device="cpu", freeze=True) -> STFBaseOptimizer:

    model = STFBaseOptimizer(drop_path_rate=0.0)
    state_dict = torch.load(path, map_location=torch.device(device))['state_dict']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)

    model.update()

    if freeze:
        for param in model.parameters():
            param.requires_grad = False

    return model

def plot_reconstruction(original_image, reconstructed_image, normal_reconstruction):

    # Plot comparisons
    fig, ax = plt.subplots(nrows=normal_reconstruction.shape[0], ncols=4, figsize=(36, 9*normal_reconstruction.shape[0]))
    for i in range(normal_reconstruction.shape[0]):
        normal_reconstruction_im = normal_reconstruction[i].detach().permute(1, 2, 0)
        reconstructed_image_im = reconstructed_image[i].detach().permute(1, 2, 0)
        diff = torch.sum(torch.abs(normal_reconstruction_im - reconstructed_image_im).detach(), -1)
        print("Difference:", torch.sum(torch.abs(normal_reconstruction_im - reconstructed_image_im)).item())
        ax[i, 0].imshow(original_image[i].permute(1, 2, 0))
        ax[i, 1].imshow(normal_reconstruction_im, vmin=0, vmax=1)
        ax[i, 2].imshow(reconstructed_image_im, vmin=0, vmax=1)
        ax[i, 3].imshow(diff, cmap='hot')
        ax[i, 0].set_title("Original")
        ax[i, 1].set_title("Reconstructed")
        ax[i, 2].set_title("Our Reconstruction")
        ax[i, 3].set_title("Difference of methods")

        cmap = cm.get_cmap("hot")
        norm = colors.Normalize(diff.min(), diff.max())

        fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax[i, 3])

    plt.show()