import torch

from our_model import OurModel

# View as trained compression
def compress_by_y_optimization_through_decoder(model: OurModel, original_image, normal_reconstruction, iterations=1000):

    '''
    Generate compression from optimizing y when decoding and reconstructing the image
    '''
    
    y, Wh, Ww = model.continious_compress_to_y(original_image)
    y_param = torch.nn.Parameter(y, requires_grad=True)

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

    return real_compress

# View as trained decrompession
def decompress_by_image_optimization_through_encoder(model: OurModel, compressed, normal_reconstruction, learning_rate=0.0001, iterations=1000):
    '''
    Reconstruct images by optimizing image to get the same y_bar as given from compression
    '''
    real_y_bar, _ = model.real_decode_to_y_bar(*compressed.values())

    # Regain images
    reconstructed_image = model.decompress(*compressed.values())['x_hat']
    reconstructed_image = torch.nn.Parameter(reconstructed_image.detach(), requires_grad=True)


    learning_rate = 0.0001
    optim = torch.optim.Adam([reconstructed_image], lr=learning_rate)

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

    return reconstructed_image


def y_bar_optimize_from_imagedecoder(model: OurModel, original_image, normal_reconstruction, learning_rate=0.001, iterations=1000):
    '''
    Reconstruct images by using encoder-decoder, then optimizing the latent representation towards the reconstruction
    '''

    standard_compression = model.compress(original_image)
    standard_y_hat, standard_z_hat = model.real_decode_to_y_bar(*standard_compression.values())
    standard_y_hat = standard_y_hat.detach()
    y_hat_param = torch.nn.Parameter(standard_y_hat, requires_grad=True)


    learning_rate = 0.0001
    optim = torch.optim.Adam([y_hat_param], lr=learning_rate)

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

    return reconstructed_image
