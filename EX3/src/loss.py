import torch 

def kld(z_mean, z_std):
    batch_size = z_mean.shape[0]
    kl = (z_std ** 2 + z_mean ** 2 - torch.log(z_std) - 0.5).sum() / batch_size
    return kl

def weighted_loss(mse, kld, mse_weight):
    assert mse_weight <= 1.0, "mse_weight should be <= 1.0"
    return (mse_weight * mse) + (1.0 - mse_weight) * kld