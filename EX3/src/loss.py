import torch 

def kld(z_mean, z_log_vars):    
    kl = -0.5 * torch.sum(1 + z_log_vars - z_mean.pow(2) - z_log_vars.exp())
    return kl