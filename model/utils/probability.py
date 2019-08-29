import sys
import torch
import numpy as np
from .convert import to_var


def normal_logpdf(x, mean, var):
    """
    Args:
        x: (Variable, FloatTensor) [batch_size, dim]
        mean: (Variable, FloatTensor) [batch_size, dim] or [batch_size] or [1]
        var: (Variable, FloatTensor) [batch_size, dim]: positive value
    Return:
        log_p: (Variable, FloatTensor) [batch_size]
    """

    pi = to_var(torch.FloatTensor([np.pi]))
    return 0.5 * torch.sum(-torch.log(2.0 * pi) - torch.log(var) - ((x - mean).pow(2) / var), dim=1)


def dirichlet_logpdf(x, alpha):
    one = to_var(torch.FloatTensor([1.0]))
    return torch.mvlgamma(torch.sum(alpha, 1), 1) - torch.sum(torch.mvlgamma(alpha, 1), 1) + \
           torch.sum((alpha - one) * torch.log(x))


def normal_kl_div(mu1, var1,
                  mu2=to_var(torch.FloatTensor([0.0])),
                  var2=to_var(torch.FloatTensor([1.0]))):
    one = to_var(torch.FloatTensor([1.0]))
    return torch.sum(0.5 * (torch.log(var2) - torch.log(var1)
                            + (var1 + (mu1 - mu2).pow(2)) / var2 - one), 1)


def dirichlet_kl_div(alpha1,
                     alpha2=to_var(torch.FloatTensor([0.0]))):
    alpha0 = to_var(torch.Tensor(alpha1.shape[1], alpha1.shape[0]))
    alpha0 = torch.transpose(alpha0.copy_(torch.sum(alpha1, 1)), 1, 0)
    try:
        return torch.mvlgamma(torch.sum(alpha1, 1), 1) - torch.mvlgamma(torch.sum(alpha2, 1), 1) - \
               torch.sum(torch.mvlgamma(alpha1, 1), 1) + torch.sum(torch.mvlgamma(alpha2, 1), 1) + \
               torch.sum((alpha1 - alpha2) * (torch.digamma(alpha1) - torch.digamma(alpha0)), 1)
    except RuntimeError:
        print(alpha0)
        print(alpha1)
        print(alpha2)
        sys.exit(-1)
