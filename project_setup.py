import torch

from compressai.models import SymmetricalTransFormer

import matplotlib.pyplot as plt
from matplotlib import cm, colors

from my_utils import pull_openimages, prepare_data_loader, load_pretrained_model, OurModel

# View as trained compression
def compress_by_y_optimization_through_decoder(model: OurModel, original_image, iterations=1000, standard_compression=None):

    '''
    Generate compression from optimizing y when decoding and reconstructing the image
    '''

    if standard_compression is None:
        standard_compression = model.compress(original_image)
    
    y, Wh, Ww = model.continious_compress_to_y(original_image)
    y_param = torch.nn.Parameter(y, requires_grad=True)

    normal_reconstruction = model.decompress(*standard_compression.values())['x_hat']

    learning_rate = 0.0001
    optim = torch.optim.Adam([y_param], lr=learning_rate)

    for i in range(iterations):

        reconstructed_image = model.continious_decompress_from_y(y_param, Wh, Ww)['x_hat']

        loss = torch.nn.functional.mse_loss(reconstructed_image, original_image)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

    real_compress = model.real_compress_y(y_param, Wh, Ww)
    reconstructed_image = model.decompress(*real_compress.values())['x_hat']
    print(f"Final, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

    return reconstructed_image, normal_reconstruction, real_compress

# View as trained decrompession
def decompress_by_image_optimization_through_encoder(model: OurModel, original_image, learning_rate=0.001, iterations=1000, standard_compression=None):
    '''
    Reconstruct images by optimizing image to get the same y_bar as given from compression
    '''

    if standard_compression is None:
        standard_compression = model.compress(original_image)

    real_y_bar, _ = model.real_decode_to_y_bar(*standard_compression.values())

    # Regain images
    reconstructed_image = model.decompress(*standard_compression.values())['x_hat']
    reconstructed_image = torch.nn.Parameter(reconstructed_image.detach(), requires_grad=True)


    learning_rate = 0.0001
    optim = torch.optim.Adam([reconstructed_image], lr=learning_rate)

    normal_reconstruction = model.decompress(*standard_compression.values())['x_hat']

    # Optimize reconstructed images
    for i in range(iterations):
        # Now feed this reconstruction back
        y_bar = model.continious_compress_to_y_bar(reconstructed_image)

        # Find error
        loss = torch.nn.functional.mse_loss(y_bar, real_y_bar) #+ torch.nn.functional.mse_loss(r_z_hat, standard_z_hat)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

    return reconstructed_image, normal_reconstruction

def optimize_y_and_y_bar(model: OurModel, original_image, iterations=1000):

    standard_compression = model.compress(original_image)
    normal_reconstruction = model.decompress(*standard_compression.values())['x_hat']

    _, _, real_compress = compress_by_y_optimization_through_decoder(model, original_image, iterations=iterations)

    reconstructed_image, _ = decompress_by_image_optimization_through_encoder(model, original_image, iterations=iterations, standard_compression=real_compress)

    return reconstructed_image, normal_reconstruction



def y_bar_optimize_from_imagedecoder(model: OurModel, original_image, learning_rate=0.001, iterations=1000):
    '''
    Reconstruct images by using encoder-decoder, then optimizing the latent representation towards the reconstruction
    '''

    standard_compression = model.compress(original_image)
    standard_y_hat, standard_z_hat = model.real_decode_to_y_bar(*standard_compression.values())
    standard_y_hat = standard_y_hat.detach()
    y_hat_param = torch.nn.Parameter(standard_y_hat, requires_grad=True)


    learning_rate = 0.0001
    optim = torch.optim.Adam([y_hat_param], lr=learning_rate)

    normal_reconstruction = model.decompress(*standard_compression.values())['x_hat']

    y_shape = [standard_z_hat.shape[2] * 4, standard_z_hat.shape[3] * 4]
    Wh, Ww = y_shape

    reconstructed_image = None

    for i in range(iterations):

        reconstructed_image = model.continious_decompress_from_y_bar(y_hat_param, Wh, Ww)

        loss = torch.nn.functional.mse_loss(reconstructed_image, original_image)

        optim.zero_grad()
        loss.backward()
        optim.step()

        print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

    return reconstructed_image, normal_reconstruction

def plot_reconstruction(original_image, reconstructed_image, normal_reconstruction):


    # Plot comparisons
    fig, ax = plt.subplots(nrows=normal_reconstruction.shape[0], ncols=4)
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



# Some defenitions
device = "cpu"
dataset_dir = "openimages"
channels = 3
test_batch_size = 2
patch_size = (256, 256)

# Prepare data and load model
#pull_openimages(traning_size=test_batch_size+1, test_size=test_batch_size, dataset_dir=dataset_dir)
test_dataloader = prepare_data_loader(patch_size=patch_size, test_batch_size=test_batch_size, device=device, dataset_dir=dataset_dir)
model = load_pretrained_model(path="compressai/pretrained/stf_0035_best.pth.tar", device=device, freeze=True)

our_batch = next(iter(test_dataloader))
our_batch = our_batch.to(device)

# Do training and plot
#reconstructed_image, normal_reconstruction, _ = compress_by_y_optimization_through_decoder(model, our_batch, iterations=10)
#reconstructed_image, normal_reconstruction = decompress_by_image_optimization_through_encoder(model, our_batch, iterations=3)
#reconstructed_image, normal_reconstruction = y_bar_optimize_from_imagedecoder(model, our_batch, iterations=3)
reconstructed_image, normal_reconstruction = optimize_y_and_y_bar(model, our_batch, iterations=1000)
plot_reconstruction(our_batch, reconstructed_image, normal_reconstruction)