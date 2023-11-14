import torch
from openimages_load import pull_openimages

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.models import SymmetricalTransFormer

import matplotlib.pyplot as plt

device = "cpu"

dataset_dir = "openimages"
test_batch_size = 2
patch_size = (256, 256)

#pull_openimages(traning_size=test_batch_size+1, test_size=test_batch_size, dataset_dir=dataset_dir)

test_transforms = transforms.Compose(
        [transforms.CenterCrop(patch_size), transforms.ToTensor()]
    )

test_dataset = ImageFolder(dataset_dir, split="test", transform=test_transforms)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=test_batch_size,
    #num_workers=1,
    shuffle=False,
    pin_memory=(device == "cuda"),
)

# Load model
#model_info = torch.load("compressai/pretrained/stf_0035_best.pth.tar", map_location=torch.device('cpu'))
model = SymmetricalTransFormer()
state_dict = torch.load("compressai/pretrained/stf_0035_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)

model.update()


with torch.no_grad():
    our_batch = next(iter(test_dataloader))
    our_batch = our_batch.to(device)
    comp_out = model.compress3(our_batch)
    out_net = model.decompress(*comp_out.values())['x_hat']
    #out_net = model(our_batch)['x_hat']

print("Jomar")
input()
# Plot comparisons
fig, ax = plt.subplots(nrows=out_net.shape[0], ncols=2)
for i in range(out_net.shape[0]):
    ax[i, 0].imshow(our_batch[i].permute(1, 2, 0))
    ax[i, 1].imshow(out_net[i].permute(1, 2, 0))
    ax[i, 0].set_title("Original")
    ax[i, 1].set_title("Reconstructed")

plt.show()