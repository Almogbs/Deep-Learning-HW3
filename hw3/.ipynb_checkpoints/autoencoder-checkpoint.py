import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        kernel_size = (5, 5)
        stride = (2, 2)
        padding = (2, 2)

        activation_layer = nn.ReLU()
        modules = []
        input_channels = [in_channels, 32, 64, 128]
        output_channels = [32, 64, 128, out_channels]

        for in_channels, out_channels in zip(input_channels[:-1], output_channels[:-1]):
            modules.append(nn.Conv2d(in_channels=in_channels,
                                     out_channels=out_channels,
                                     kernel_size=kernel_size,
                                     padding=padding,
                                     stride=stride
                                    )
            )
            modules.append(nn.BatchNorm2d(num_features=out_channels))
            modules.append(activation_layer)

        modules.append(nn.Conv2d(in_channels=input_channels[-1],
                                 out_channels=output_channels[-1],
                                 kernel_size=kernel_size,
                                 padding=padding
                                )
        )

        self.cnn = nn.Sequential(*modules)

    
    def forward(self, x):
        return self.cnn(x)


class DecoderCNN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        kernel_size = (5, 5)
        stride = (2, 2)
        padding = (2, 2)
        output_padding = (1, 1)

        activation_layer = nn.ReLU()
        modules = []
        input_channels = [128, 64, 32, out_channels]
        output_channels = [in_channels, 128, 64, 32]

        modules.append(nn.ConvTranspose2d(in_channels=output_channels[0],
                                          out_channels=input_channels[0],
                                          kernel_size=kernel_size,
                                          padding=padding
                                         )
        )

        for out_channels, in_channels in zip(input_channels[1:], output_channels[1:]):
            modules.append(activation_layer)
            modules.append(nn.BatchNorm2d(num_features=in_channels))
            modules.append(nn.ConvTranspose2d(in_channels=in_channels,
                                              out_channels=out_channels,
                                              kernel_size=kernel_size,
                                              padding=padding,
                                              stride=stride,
                                              output_padding=output_padding
                                             )
            )

        self.cnn = nn.Sequential(*modules)

    def forward(self, h):
        # Tanh to scale to [-1, 1] (same dynamic range as original images).
        return torch.tanh(self.cnn(h))


class VAE(nn.Module):
    def __init__(self, features_encoder, features_decoder, in_size, z_dim):
        """
        :param features_encoder: Instance of an encoder the extracts features
        from an input.
        :param features_decoder: Instance of a decoder that reconstructs an
        input from i t's features.
        :param in_size: The size of one input (without batch dimension).
        :param z_dim: The latent space dimension.
        """
        super().__init__()
        self.features_encoder = features_encoder
        self.features_decoder = features_decoder
        self.z_dim = z_dim

        self.features_shape, n_features = self._check_features(in_size)
        self.mean_fc = nn.Linear(n_features, z_dim, bias=True)
        self.var_fc = nn.Linear(n_features, z_dim, bias=True)
        self.latent_fc = nn.Linear(z_dim, n_features, bias=True)

    def _check_features(self, in_size):
        device = next(self.parameters()).device
        with torch.no_grad():
            # Make sure encoder and decoder are compatible
            x = torch.randn(1, *in_size, device=device)
            h = self.features_encoder(x)
            xr = self.features_decoder(h)
            assert xr.shape == x.shape
            # Return the shape and number of encoded features
            return h.shape[1:], torch.numel(h) // h.shape[0]

    def encode(self, x):
        h = self.features_encoder(x).reshape(x.shape[0], -1)
        mu, log_sigma2 = self.mean_fc(h), self.var_fc(h)
        
        std = torch.exp(0.5 * log_sigma2)
        eps = torch.randn_like(std)
        z = mu + eps * std

        return z, mu, log_sigma2

    def decode(self, z):
        h = self.latent_fc(z).reshape(-1, *self.features_shape)
        x_rec = self.features_decoder(h)
        
        return torch.tanh(x_rec)

    def sample(self, n):
        samples = []
        device = next(self.parameters()).device
        with torch.no_grad():
            input = torch.randn((n, self.z_dim), device=device)
            samples = self.decode(input)

        samples = [s.detach().cpu() for s in samples]
        return samples

    def forward(self, x):
        z, mu, log_sigma2 = self.encode(x)
        return self.decode(z), mu, log_sigma2


def vae_loss(x, xr, z_mu, z_log_sigma2, x_sigma2):
    """
    Point-wise loss function of a VAE with latent space of dimension z_dim.
    :param x: Input image batch of shape (N,C,H,W).
    :param xr: Reconstructed (output) image batch.
    :param z_mu: Posterior mean (batch) of shape (N, z_dim).
    :param z_log_sigma2: Posterior log-variance (batch) of shape (N, z_dim).
    :param x_sigma2: Likelihood variance (scalar).
    :return:
        - The VAE loss
        - The data loss term
        - The KL divergence loss term
    all three are scalars, averaged over the batch dimension.
    """
    N, C, H, W = x.size()
    d_x = C * H * W
    d_z = z_mu.shape[1]

    inner_data_loss = x.reshape(x.shape[0], -1) - xr.reshape(xr.shape[0], -1)
    data_loss = torch.mean(torch.sum(inner_data_loss ** 2, dim=1)) 
    data_loss /= (x_sigma2 * d_x)
    
    kldiv_loss_1 = torch.sum(torch.exp(z_log_sigma2), dim=1)
    kldiv_loss_2 = torch.sum(z_mu ** 2, dim=1)
    kldiv_loss_3 = d_z
    kldiv_loss_4 = torch.sum(z_log_sigma2, dim=1)
    kldiv_loss = torch.mean(kldiv_loss_1 + kldiv_loss_2 - (kldiv_loss_3 + kldiv_loss_4))

    return kldiv_loss + data_loss, data_loss, kldiv_loss



