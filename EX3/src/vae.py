import torch

class Encoder(torch.nn.Module):

    def __init__(self,*, input_dim=28, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.flatten = torch.nn.Flatten()
        self.linear1 = torch.nn.Linear(self.input_dim * self.input_dim , 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.mean = torch.nn.Linear(256, self.latent_dim)
        self.log_var = torch.nn.Linear(256, self.latent_dim)
        self.activation = torch.nn.ReLU()


    def forward(self, x):
        x_flat = self.flatten(x)
        
        z = self.linear1(x_flat)
        z = self.activation(z)

        z = self.linear1(z)
        z = self.activation(z)

        means = self.mean(z)
        log_vars = self.log_var(z)
        stds = torch.exp(0.5 * log_vars)
        random_gaussian = torch.normal(0.0, 1.0, size=(x.shape[0], self.latent_dim))

        z = means + stds * random_gaussian

        return z, means, stds
    

class Decoder(torch.nn.Module):

    def __init__(self,*, latent_dim, output_dim=28):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.linear1 = torch.nn.Linear(self.latent_dim , 256)
        self.linear2 = torch.nn.Linear(256, 256)
        self.mean = torch.nn.Linear(256, self.output_dim * self.output_dim)
        self.log_var = torch.zeros((1,1), requires_grad=True)

        self.activation = torch.nn.ReLU()


    def forward(self, z):        
        x = self.linear1(z)
        x = self.activation(x)

        x = self.linear1(x)
        x = self.activation(x)

        means = self.mean(x)
        log_var = self.log_var(x)
        random_gaussian = torch.normal(0.0, 1.0, size=(x.shape[0],  self.output_dim * self.output_dim))

        x = means + torch.exp(0.5 * log_var) * random_gaussian

        return x, means
    

class VAE(torch.nn.Module):

    def __init__(self,*, input_dim=28, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim=self.input_dim, latent_dim=self.latent_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim, output_dim=self.input_dim)

    def forward(self, x):
        z, z_means, z_stds  = self.encoder(x)
        x_pred, x_pred_means = self.decoder(z)

        return z_means, z_stds, x_pred, x_pred_means