import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from loss import kld
import math
from tqdm import tqdm

@dataclass(kw_only=True)
class VAEConfig:
    input_dim: int | None
    latent_dim: int
    encoder_layers: list[int]
    decoder_layers: list[int]
    learning_rate: float
    batch_size: int
    epochs: int
    visualization_interval: int | None


class Encoder(torch.nn.Module):

    def __init__(self,*, input_dim=28 * 28, latent_dim, hidden_layer_sizes=[256,256]):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.hidden_layers = []
        prev_input = self.input_dim
        for units in hidden_layer_sizes:
            self.hidden_layers.append(torch.nn.Linear(prev_input, units))
            prev_input = units

        self.mean = torch.nn.Linear(prev_input, self.latent_dim)
        self.log_var = torch.nn.Linear(prev_input, self.latent_dim)
        self.activation = torch.nn.ReLU()


    def forward(self, x):

        z = x
        for layer in self.hidden_layers:
            z = self.activation(layer(z))

        means = self.mean(z)
        log_vars = self.log_var(z)
        stds = torch.exp(0.5 * log_vars)

        random_gaussian = torch.randn_like(stds)
        z = means + stds * random_gaussian

        return z, means, log_vars
    

class Decoder(torch.nn.Module):

    def __init__(self,*, latent_dim, output_dim=28 * 28, hidden_layer_sizes=[256,256]):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.hidden_layers = []
        prev_input = self.latent_dim
        for units in hidden_layer_sizes:
            self.hidden_layers.append(torch.nn.Linear(prev_input, units))
            prev_input = units

        self.sigmoid = torch.nn.Sigmoid()
        self.mean = torch.nn.Linear(prev_input, self.output_dim)
        self.log_var = torch.rand((1,1), requires_grad=True)

        self.activation = torch.nn.ReLU()


    def forward(self, z):        
        x = z
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        x_means = self.mean(x)
        x_stds = torch.exp(0.5 * self.log_var)
        random_gaussian = torch.randn_like(x_means)

        x_pred = self.sigmoid(x_means + random_gaussian * x_stds)
        return x_pred
    

class VAE(torch.nn.Module):

    def __init__(self,*, input_dim=28 * 28, latent_dim, encoder_layer_sizes=[256,256], decoder_layer_sizes=[256,256]):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim=self.input_dim, latent_dim=self.latent_dim, hidden_layer_sizes=encoder_layer_sizes)
        self.decoder = Decoder(latent_dim=self.latent_dim, output_dim=self.input_dim, hidden_layer_sizes=decoder_layer_sizes)

    def forward(self, x):
        z, z_means, z_log_vars  = self.encoder(x)
        x_pred = self.decoder(z)

        return z_means, z_log_vars, x_pred
    

class VAETrainer:
    def __init__(self, config: VAEConfig, train_set, test_set):
        self.config = config
        self.VAE = VAE(input_dim=config.input_dim or None, latent_dim=config.latent_dim,
                        encoder_layer_sizes=config.encoder_layers, decoder_layer_sizes=config.decoder_layers)
        
        self.optim = torch.optim.Adam(self.VAE.parameters(), lr=config.learning_rate)

        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        self.test_set = test_set
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mse_loss = torch.nn.MSELoss(reduction='sum')
        self.kld_loss = kld
        self.visualization_interval = self.config.visualization_interval
    
    def train(self):
        """call after initialization to train the VAE model"""
        epoch_train_losses = []
        epoch_test_losses = []
        for epoch in range(self.config.epochs):
            # Process tqdm bar
            batch_bar = tqdm(total=len(self.train_loader), 
                        leave=True, position=0, desc="Train")

            train_loss = 0.0
            mse_loss_total = 0.0
            kld_loss_total = 0.0
            for i, batch in enumerate(self.train_loader):
                
                self.optim.zero_grad()
                x = batch[0].to(self.device)
                z_means, z_log_vars, x_pred = self.VAE(x) 
                
                loss_mse = self.mse_loss(x_pred, x)
                mse_loss_total += loss_mse.item()

                loss_kld = self.kld_loss(z_means, z_log_vars)
                kld_loss_total += loss_kld.item()

                loss = loss_mse + loss_kld
                
                loss.backward()
                self.optim.step()

                train_loss += loss.item()
                
                batch_bar.set_postfix(
                    epoch = f"{epoch + 1}",
                    loss = f"{train_loss/(i+1):.4f}",
                    mse_loss = f"{loss_mse:.4f}",
                    kl_loss = f"{loss_kld:.4f}",
                    lr = f"{self.optim.param_groups[0]['lr']:.4f}"
                )
                
                batch_bar.update()
                torch.cuda.empty_cache()
            
            batch_bar.close()
            train_loss /= len(self.train_loader)
            epoch_train_losses.append(train_loss)
            
            test_loss = self.get_test_loss(z_means, z_log_vars)
            epoch_test_losses.append(test_loss)

            if self.visualization_interval is not None and (epoch + 1) % self.visualization_interval == 0:
                self.reconstruct_15_images()
                self.generate_15_images()

        epoch_list = list(range(1, self.config.epochs + 1))
        plt.title("Training ELBO Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_list, epoch_train_losses)
        plt.show()

        plt.title("Testing ELBO Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.plot(epoch_list, epoch_test_losses)
        plt.show()

    def reconstruct_15_images(self):
        """
        Used only in cases where the dataset consists of images.
        plots 15 random samples from the test set and their reconstructions
        """
        with torch.no_grad():
            fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(10, 5))
            fig.suptitle("15 Reconstructed Samples")
            test_loader =  DataLoader(self.test_set, batch_size=1, shuffle=True)
            sample_iter = iter(test_loader)
            for i in range(5):
                for j in range(0, 6, 2):
                    index = i * 6 + j
                    sample = next(sample_iter)
                    x = sample[0].to(self.device)
                    _, _, x_pred = self.VAE(x) 
                    dim = int(math.sqrt(self.config.input_dim))
                    x_2d = (x.view(-1, dim, dim) * 255).squeeze()
                    x_pred_2d = (x_pred.view(-1, dim, dim) * 255).squeeze()

                    axes[i][j].imshow(x_2d)
                    axes[i][j].axis('off')

                    axes[i][j+1].imshow(x_pred_2d)
                    axes[i][j+1].axis('off')

                    if i == 0:
                        axes[i][j].title.set_text("Sample")
                        axes[i][j + 1].title.set_text("Reconstruction")
            
            plt.show()
    
    def generate_15_images(self):
        """
        Used only in cases where the dataset consists of images.
        plots 15 randomly generated images.
        """
        with torch.no_grad():
            fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(10, 5))
            fig.suptitle("15 Generated Samples")
            for i in range(3):
                for j in range(5):
                    z = torch.randn((1, self.config.latent_dim))
                    x_pred = self.VAE.decoder(z)
                    dim = int(math.sqrt(self.config.input_dim))
                    x_pred_2d = (x_pred.view(-1, dim, dim) * 255).squeeze()

                    axes[i][j].imshow(x_pred_2d)
                    axes[i][j].axis('off')
            plt.show()

    def predict_test(self):
        """Reconstructs the test set and returns the reconstructions."""
        test_loader = DataLoader(self.test_set, batch_size=len(self.test_set), shuffle=False)
        with torch.no_grad():
            sample_iter = iter(test_loader)
            test_samples = next(sample_iter)
            x = test_samples[0].to(self.device)
            _, _, x_pred = self.VAE(x) 
            return x_pred
        
    def get_test_loss(self, z_means, z_log_vars):
        test_samples = self.test_set.tensors[0]
        test_reconstructions = self.predict_test()
        loss_mse = self.mse_loss(test_reconstructions, test_samples).item()

        loss_kld = self.kld_loss(z_means, z_log_vars).item()

        return (loss_mse + loss_kld) / len(test_samples)


    def get_n_generated_samples(self, n):
        """Generates n random samples."""
        with torch.no_grad():
            z = torch.randn((n, self.config.latent_dim))
            x_pred = self.VAE.decoder(z)

            return x_pred

