import torch
from .stf import SymmetricalTransFormer
from compressai.ops import ste_round
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.utils.eval_model import psnr, rmse_and_snr
import wandb

class STFBaseOptimizer(SymmetricalTransFormer):

    def eval(self):
        '''
        This model actually needs to keep track of gradients when doing evaluation
        '''

        for param in self.parameters():
            param.requires_grad = False

        return self
    

    def continious_compress_to_y_bar(self, x):

        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)

        y = x
        C = self.embed_dim * 8
        y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)

        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh*Ww, C)

        return y_hat
    
    def real_decode_to_y_bar(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        Wh, Ww = y_shape
        C = self.embed_dim * 8

        y_string = strings[0][0]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        for slice_index in range(self.num_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(mu.shape)
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)


            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh * Ww, C)

        return y_hat, z_hat
    
    def continious_compress_to_y(self, x):
        x = self.patch_embed(x)

        Wh, Ww = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)
        for i in range(self.num_layers):
            layer = self.layers[i]
            x, Wh, Ww = layer(x, Wh, Ww)
        y = x

        return y, Wh, Ww
    
    def real_compress_y(self, y, Wh, Ww):

        C = self.embed_dim * 8
        y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])

            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)

        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}
    
    def continious_decompress_from_y(self, y, Wh, Ww):

        C = self.embed_dim * 8
        y = y.view(-1, Wh, Ww, C).permute(0, 3, 1, 2).contiguous()
        y_shape = y.shape[2:]

        z = self.h_a(y)
        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset

        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            support_slices = (y_hat_slices if self.max_support_slices < 0 else y_hat_slices[:self.max_support_slices])
            mean_support = torch.cat([latent_means] + support_slices, dim=1)
            mu = self.cc_mean_transforms[slice_index](mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([latent_scales] + support_slices, dim=1)
            scale = self.cc_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)

            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.lrp_transforms[slice_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)

        y_hat = y_hat.permute(0, 2, 3, 1).contiguous().view(-1, Wh*Ww, C)
        for i in range(self.num_layers):
            layer = self.syn_layers[i]
            y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        x_hat = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }
    

    def continious_decompress_from_y_bar(self, y_hat, Wh, Ww):

        for i in range(self.num_layers):
            layer = self.syn_layers[i]
            y_hat, Wh, Ww = layer(y_hat, Wh, Ww)

        x_hat = self.end_conv(y_hat.view(-1, Wh, Ww, self.embed_dim).permute(0, 3, 1, 2).contiguous())

        return x_hat
    
    # View as trained compression
    def optimized_compress(self, original_image, iterations=1000, normal_reconstruction=None, verbose=False,
                            wandb_log=False, wandb_project=None, log_every=100):

        '''
        Generate compression from optimizing y when decoding and reconstructing the image
        '''
        
        y, Wh, Ww = self.continious_compress_to_y(original_image)
        y_param = torch.nn.Parameter(y, requires_grad=True)

        if wandb_log:
            wandb.init(project=wandb_project)


        learning_rate = 0.0001
        optim = torch.optim.Adam([y_param], lr=learning_rate)
        psnr_scores = []
        rmse = []
        ms_snr = []
        for i in range(iterations):

            reconstructed_image = self.continious_decompress_from_y(y_param, Wh, Ww)['x_hat']

            loss = torch.nn.functional.mse_loss(reconstructed_image, original_image)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            if wandb_log:
                if i % log_every == 0:
                    psnr_scores.append(psnr(original_image, reconstructed_image))
                    rmse.append(rmse_and_snr(original_image, reconstructed_image)[0])
                    ms_snr.append(rmse_and_snr(original_image, reconstructed_image)[1])
                    wandb.log({"loss": loss})

            if verbose:
                print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

        real_compress = self.real_compress_y(y_param, Wh, Ww)

        if verbose:
            reconstructed_image = super().decompress(*real_compress.values())['x_hat']
            print(f"Final, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

        if wandb_log:
            data = [[log_every*x, y, z, w] for (x, y, z, w) in zip(range(1, len(psnr_scores)+1), psnr_scores, rmse, ms_snr)]
            table = wandb.Table(data=data, columns=["Iteration", "PSNR", "RMSE", "MS-SNR"])

            wandb.log({
                "psnr_plot": wandb.plot.line(table, "Iteration", "PSNR", title="PSNR compress"),
                "rmse_plot": wandb.plot.line(table, "Iteration", "RMSE", title="RMSE compress"),
                "ms_snr_plot": wandb.plot.line(table, "Iteration", "MS-SNR", title="MS-SNR compress")


            })
       
        wandb_rec = [wandb.Image(img, caption="Reconsruction {}".format(i+1)) for i, img in enumerate(reconstructed_image)]
        wandb.log({"Reconstructions": wandb_rec})

        wandb_original = [wandb.Image(img, caption="Original {}".format(i+1)) for i, img in enumerate(original_image)]
        wandb.log({"Original images": wandb_original})




        print(reconstructed_image.shape)
        return real_compress

    # View as trained decrompession
    def optimized_decompress(self, strings, shape, learning_rate=0.0001, iterations=1000, normal_reconstruction=None, verbose=False,
                            wandb_log=False, wandb_project=None, log_every=100):
        '''
        Reconstruct images by optimizing image to get the same y_bar as given from compression
        '''

        if wandb_log:
            wandb.init(project=wandb_project)

        real_y_bar, _ = self.real_decode_to_y_bar(strings, shape)

        # Regain images
        reconstructed_image = super().decompress(strings, shape)['x_hat']
        reconstructed_image = torch.nn.Parameter(reconstructed_image.detach(), requires_grad=True)


        learning_rate = 0.0001
        optim = torch.optim.Adam([reconstructed_image], lr=learning_rate)

        psnr_scores_de = []
        rmse_de = []
        ms_snr_de = []

        # Optimize reconstructed images
        for i in range(iterations):
            # Now feed this reconstruction back
            y_bar = self.continious_compress_to_y_bar(reconstructed_image)

            # Find error
            loss = torch.nn.functional.mse_loss(y_bar, real_y_bar) #+ torch.nn.functional.mse_loss(r_z_hat, standard_z_hat)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if wandb_log:
                if i % log_every == 0:
                    psnr_scores_de.append(psnr(real_y_bar, y_bar))
                    rmse_de.append(rmse_and_snr(real_y_bar, y_bar)[0])
                    ms_snr_de.append(rmse_and_snr(real_y_bar, y_bar)[1])
                    wandb.log({"loss": loss})

            if verbose:
                print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

        if wandb_log:
            data_de = [[log_every*x, y, z, w] for (x, y, z, w) in zip(
                    range(1, len(psnr_scores_de)+1), psnr_scores_de, rmse_de, ms_snr_de)]
            table = wandb.Table(data=data_de, columns=["Iteration D", "PSNR D", "RMSE D", "MS-SNR D"])

            wandb.log({
                "psnr_plot_de": wandb.plot.line(table, "Iteration D", "PSNR D", title="PSNR decompress"),
                "rmse_plot_de": wandb.plot.line(table, "Iteration D", "RMSE D", title="RMSE decompress"),
                "ms_snr_plot_de": wandb.plot.line(table, "Iteration D", "MS-SNR D", title="MS-SNR decompress")
            })

        wandb_rec = [wandb.Image(img, caption="Reconsruction {}".format(i+1)) for i, img in enumerate(reconstructed_image)]
        wandb.log({"Reconstructions": wandb_rec})

        return {"x_hat": reconstructed_image}
    
    def y_bar_optimize_from_imagedecoder(self, strings, shape, original_image=None, learning_rate=0.001, iterations=1000, normal_reconstruction=None, verbose=False):
        '''
        Reconstruct images by using encoder-decoder, then optimizing the latent representation towards the reconstruction
        '''
        #standard_compression = self.compress(original_image)
        standard_y_hat, standard_z_hat = self.real_decode_to_y_bar(strings, shape)
        standard_y_hat = standard_y_hat.detach()
        y_hat_param = torch.nn.Parameter(standard_y_hat, requires_grad=True)


        learning_rate = 0.0001
        optim = torch.optim.Adam([y_hat_param], lr=learning_rate)

        y_shape = [standard_z_hat.shape[2] * 4, standard_z_hat.shape[3] * 4]
        Wh, Ww = y_shape

        reconstructed_image = None

        for i in range(iterations):

            reconstructed_image = self.continious_decompress_from_y_bar(y_hat_param, Wh, Ww)

            loss = torch.nn.functional.mse_loss(reconstructed_image, original_image)

            optim.zero_grad()
            loss.backward()
            optim.step()

            if verbose:
                print(f"Iteration {i+1}, loss: {loss.item()}, difference: {torch.sum(torch.abs(normal_reconstruction - reconstructed_image)).item()}")

        return {"x_hat": reconstructed_image}
    

    
class STFCompressOptimizer(STFBaseOptimizer):

    def compress(self, x):
        with torch.enable_grad():
            return self.optimized_compress(x)
    
class STFDecompressOptimizer(STFBaseOptimizer):

    def decompress(self, strings, shape):
        with torch.enable_grad():
            return self.optimized_decompress(strings, shape)

class STFFullOptimizer(STFBaseOptimizer):

    def compress(self, x):
        with torch.enable_grad():
            return self.optimized_compress(x)
    
    def decompress(self, strings, shape):
        with torch.enable_grad():
            return self.optimized_decompress(strings, shape)
        
class STFDemonstrateNoQuantization(STFBaseOptimizer):
    
    def compress(self, x):
        self.original_image = x
        super().compress(x)

    def decompress(self, strings, shape):
        with torch.enable_grad():
            return self.y_bar_optimize_from_imagedecoder(strings, shape, self.original_image)
        