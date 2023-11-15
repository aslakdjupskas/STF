import torch
from openimages_load import pull_openimages

from torch.utils.data import DataLoader
from torchvision import transforms

from compressai.datasets import ImageFolder
from compressai.models import SymmetricalTransFormer

import matplotlib.pyplot as plt
from matplotlib import cm, colors

device = "cpu"

dataset_dir = "openimages"
channels = 3
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
model = SymmetricalTransFormer(drop_path_rate=0.0)
state_dict = torch.load("compressai/pretrained/stf_0035_best.pth.tar", map_location=torch.device('cpu'))['state_dict']
state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
model.to(device)

model.update()
#model.train()

#class myImage()

class myImage(torch.nn.Module):

    def __init__(self, image):
        super(myImage, self).__init__()
        self.image = image.detach().requires_grad_(True)
        self.image = torch.nn.Parameter(self.image, requires_grad=True)

    def forward(self):
        return self.image

#with torch.no_grad():

our_batch = next(iter(test_dataloader))
our_batch = our_batch.to(device)

'''normal_compression = model.compress(our_batch)

y_hat_decoded, z_hat_decoded = model.decode_latent(*normal_compression.values())
y_hat_compressed, z_hat_compressed = model.continious_compress2(our_batch)



print("y_hat_decoded", y_hat_decoded.shape, "y_hat_compressed", y_hat_compressed.shape)
print("z_hat_decoded", z_hat_decoded.shape, "z_hat_compressed", z_hat_compressed.shape)

print("y_hat_decoded", y_hat_decoded[1][2][100], "y_hat_compressed", y_hat_compressed[1][2][100])
print("z_hat_decoded", z_hat_decoded[1][1][2][3], "z_hat_compressed", z_hat_compressed[1][1][2][3])'''



#model.decompress(*normal_compression.values())

#print(model(our_batch)["likelihoods"]["y"].shape)
#print("z_hat_decoded", z_hat_decoded[0][)
#comp_out = model.compress3(our_batch)
#out_net = model.decompress(*comp_out.values())['x_hat']
#out_net = model(our_batch)['x_hat']

# Don' train model itself
for param in model.parameters():
    param.requires_grad = False

# Compress image and save y_hat and z_hat
standard_compresseion = model.compress(our_batch)
standard_y_hat, standard_z_hat = model.decode_latent(*standard_compresseion.values())
standard_y_hat = standard_y_hat.detach()
standard_z_hat = standard_z_hat.detach()

# Regain images
reconstructed_image = model.decompress(*standard_compresseion.values())['x_hat']
reconstructed_image = torch.nn.Parameter(reconstructed_image.detach(), requires_grad=True)

#mse_loss_fn = torch.nn.MSELoss()
learning_rate = 0.0001
optim = torch.optim.Adam([reconstructed_image], lr=learning_rate)

normal_reconstruction = model.decompress(*standard_compresseion.values())['x_hat']

better_image = None
for i in range(10):
    # Now feed this reconstruction back
    r_y_hat, r_z_hat = model.continious_compress2(reconstructed_image)

    # Find error
    loss = torch.nn.functional.mse_loss(r_y_hat, standard_y_hat) + torch.nn.functional.mse_loss(r_z_hat, standard_z_hat)

    optim.zero_grad()
    loss.backward()
    optim.step()

    print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")



#print("Jomar")
#input()
# Plot comparisons
fig, ax = plt.subplots(nrows=normal_reconstruction.shape[0], ncols=4)
#print(normal_reconstruction.shape[0])
for i in range(normal_reconstruction.shape[0]):
    #print(i)
    normal_reconstruction_im = normal_reconstruction[i].detach().permute(1, 2, 0)
    reconstructed_image_im = reconstructed_image[i].detach().permute(1, 2, 0)
    diff = torch.sum(torch.abs(normal_reconstruction_im - reconstructed_image_im).detach(), -1)
    #print(diff.shape)
    print("Difference:", torch.sum(torch.abs(normal_reconstruction_im - reconstructed_image_im)).item())
    ax[i, 0].imshow(our_batch[i].permute(1, 2, 0))
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
    #plt.colorbar(ax=ax[i, 3])

plt.show()