import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)


class LatentModel(nn.Module):
    """LatentModel module"""
    def __init__(self):
        super(LatentModel, self).__init__()

    def reparameterize(self, mu, logvar, nsamples=1):
        """sample from posterior Gaussian family
        Args:
            mu: Tensor
                Mean of gaussian distribution with shape (batch, nz)
            logvar: Tensor
                logvar of gaussian distibution with shape (batch, nz)
        Returns: Tensor
            Sampled z with shape (batch, nsamples, nz)
        """
        batch_size, nz = mu.size()
        std = logvar.mul(0.5).exp()

        mu_expd = mu.unsqueeze(1).expand(batch_size, nsamples, nz)
        std_expd = std.unsqueeze(1).expand(batch_size, nsamples, nz)

        eps = torch.zeros_like(std_expd).normal_()

        return mu_expd + torch.mul(eps, std_expd)
    
    def vae_latent_forward(self, hidden_states):
        mu, logvar = hidden_states.chunk(2, -1)   # [bsz x (2*latent_size)] -> two [bsz x latent_size] vectors
        latent_z = self.reparameterize(mu, logvar, nsamples=1)
        latent_z = latent_z.squeeze(1)
        return latent_z, mu, logvar
    
    def kl_loss(self, mean1, logvar1, mean2, logvar2, dim_target_kl):
        # loss_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1)
        # mean1 and logvar1 are from posterior
        # mean2 and logvar2 are from prior
        # TODO what is dim_target_kl
        exponential = logvar1 - logvar2 - torch.pow(mean1 - mean2, 2) / logvar2.exp() - torch.exp(logvar1 - logvar2) + 1
        loss_kl = -0.5 * torch.sum(exponential, tuple(range(1, len(exponential.shape))))

        kl_mask = (loss_kl > dim_target_kl).float()
        # TODO double check here: how the average should be done
        # print("Loss KL before mean", kl_mask * loss_kl, loss_kl.shape)
        loss_kl = (kl_mask * loss_kl).mean()
        # print("Loss KL", loss_kl, loss_kl.shape, loss_kl)
        return loss_kl
    
    def compute_loss(self):
        pass


def convert_mask(mask: torch.Tensor, dtype: torch.dtype):
    """
    Convert attention_mask from `[0/1]` to `[-inf/0]`.
    """

    inverted_mask = 1.0 - mask.to(dtype)

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


class AverageSelfAttention(nn.Module):
    """This fuction is modified from https://github.com/fangleai/TransformerCVAE/blob/master/model.py#L58"""
    def __init__(self, attention_size):
        super(AverageSelfAttention, self).__init__()
        w = torch.empty(attention_size)
        nn.init.normal_(w, std=0.02)
        self.attention_weights = nn.Parameter(w)
        self.softmax = nn.Softmax(dim=-1)
        self.non_linearity = GELU()
    

    def forward(self, inputs, attention_mask=None):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        if attention_mask is not None:
            attention_mask = convert_mask(attention_mask, dtype=inputs.dtype)
            scores = scores + attention_mask

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################
        scores = self.softmax(scores)

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze(1)

        return representations, scores