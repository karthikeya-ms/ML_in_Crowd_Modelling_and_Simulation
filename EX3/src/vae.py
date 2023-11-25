import torch
from dataclasses import dataclass
from torch.utils.data import DataLoader
from src.loss import weighted_loss, kld
import tqdm
@dataclass(kw_only=True)
class VAEConfig:
    input_dim: int | None
    latent_dim: int
    encoder_layers: list[int]
    decoder_layers: list[int]
    learning_rate: float
    batch_size: int
    epochs: int
    mse_weight: int


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

        random_gaussian = torch.normal(0.0, 1.0, size=(z.shape[0], self.latent_dim))

        z = means + stds * random_gaussian

        return z, means, stds
    

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

        self.mean = torch.nn.Linear(prev_input, self.output_dim)
        self.log_var = torch.zeros((1,1), requires_grad=True)

        self.activation = torch.nn.ReLU()


    def forward(self, z):        
        x = z
        for layer in self.hidden_layers:
            x = self.activation(layer(x))

        means = self.mean(x)
        random_gaussian = torch.normal(0.0, 1.0, size=(x.shape[0],  self.output_dim))

        x = means + torch.exp(0.5 * self.log_var) * random_gaussian
        return x, means
    

class VAE(torch.nn.Module):

    def __init__(self,*, input_dim=28 * 28, latent_dim, encoder_layer_sizes=[256,256], decoder_layer_sizes=[256,256]):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.encoder = Encoder(input_dim=self.input_dim, latent_dim=self.latent_dim, hidden_layer_sizes=encoder_layer_sizes)
        self.decoder = Decoder(latent_dim=self.latent_dim, output_dim=self.input_dim, hidden_layer_sizes=decoder_layer_sizes)

    def forward(self, x):
        z, z_means, z_stds  = self.encoder(x)
        x_pred, x_pred_means = self.decoder(z)

        return z_means, z_stds, x_pred, x_pred_means
    

class VAETrainer:
    def __init__(self, config: VAEConfig, train_set, test_set):
        self.config = config
        
        assert config.mse_weight <= 1.0, "mse_weight should be <= 1.0"
        self.mse_weight = config.mse_weight

        self.VAE = VAE(input_dim=config.input_dim or None, latent_dim=config.latent_dim,
                        encoder_layer_sizes=config.encoder_layers, decoder_layer_sizes=config.decoder_layers)
        
        self.optim = torch.optim.Adam(self.VAE.parameters(), lr=config.learning_rate)

        self.train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mse_loss = torch.nn.MSELoss()
        self.kld_loss = kld
    
    def train(self):
        # Process tqdm bar
        batch_bar = tqdm(total=len(self.train_loader), 
                        leave=False, position=0, desc="Train")

        train_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            
            self.optim.zero_grad()
            x = batch[0].to(self.device)
            x = x.view(-1)
            
            with torch.cuda.amp.autocast():
                # get reconstruct image
                z_means, z_stds, x_pred, x_pred_means = self.VAE(x) 
                
                loss_mse = self.mse_loss(x_pred, x)
                loss_kld = self.kld_loss(z_means, z_stds)
                
                loss = (self.mse_weight * loss_mse) + (1.0 - self.mse_weight) * loss_kld
                
            loss.backward()
            self.optim.step()

            train_loss += loss.item()
            
            batch_bar.set_postfix(
                loss = f"{train_loss/(i+1):.4f}",
                mse_loss = f"{loss_mse:.4f}",
                kl_loss = f"{loss_kld:.4f}",
                lr = f"{self.optim.param_groups[0]['lr']:.4f}"
            )
            
            batch_bar.update()
            torch.cuda.empty_cache()
            del z_means, z_stds, x_pred, x_pred_mean
        
        batch_bar.close()
        train_loss /= len(self.train_loader)

        return train_loss