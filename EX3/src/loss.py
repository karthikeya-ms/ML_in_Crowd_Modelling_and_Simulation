import torch 

def kld(z_mean, z_log_vars):
    """
    Calculate the kl loss of the posterior distribution.

    :param z_means: the predicted means of the posterior.
    :param z_log_vars: the predicted log variance of the posterior.
    :return the KL loss of the posterior.
    """    
    kl = -0.5 * torch.sum(1 + z_log_vars - z_mean.pow(2) - z_log_vars.exp())
    return kl