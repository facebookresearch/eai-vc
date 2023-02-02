import torch
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self,
        encoder_layer_sizes,
        latent_size,
        decoder_layer_sizes,
        conditional=False,
        cond_size=0,
    ):
        super().__init__()

        if conditional:
            assert cond_size > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.encoder = Encoder(encoder_layer_sizes, latent_size, conditional, cond_size)
        self.decoder = Decoder(decoder_layer_sizes, latent_size, conditional, cond_size)

    def forward(self, x, c=None):
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

    def inference(self, z, c=None):
        recon_x = self.decoder(z, c)

        return recon_x


class Encoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, cond_size):
        super().__init__()

        self.conditional = conditional
        if self.conditional:
            layer_sizes[0] += cond_size

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        self.linear_means = nn.Linear(layer_sizes[-1], latent_size)
        self.linear_log_var = nn.Linear(layer_sizes[-1], latent_size)

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        means = self.linear_means(x)
        log_vars = self.linear_log_var(x)

        return means, log_vars


class Decoder(nn.Module):
    def __init__(self, layer_sizes, latent_size, conditional, cond_size):
        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + cond_size
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(
            zip([input_size] + layer_sizes[:-1], layer_sizes)
        ):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size)
            )
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, z, c):
        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x
