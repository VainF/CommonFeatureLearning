import torch
import torch.nn as nn
import torch.nn.functional as F
from .mmd import calc_mmd

class CFLoss(nn.Module):
    """ Common Feature Learning Loss
        CF Loss = MMD + beta * MSE
    """
    def __init__(self, sigmas=[0.001, 0.01, 0.05, 0.1, 0.2, 1, 2], normalized=True, beta=1.0):
        super(CFLoss, self).__init__()
        self.sigmas = sigmas
        self.normalized = normalized
        self.beta = beta

    def forward(self, hs, ht, ft_, ft):
        mmd_loss = 0.0
        mse_loss = 0.0
        for ht_i in ht:
            mmd_loss += calc_mmd(hs, ht_i, sigmas=self.sigmas, normalized=self.normalized)
        for i in range(len(ft_)):
            mse_loss += F.mse_loss(ft_[i], ft[i])
        
        return mmd_loss + self.beta*mse_loss

class SoftCELoss(nn.Module):
    """ KD Loss Function (CrossEntroy for soft targets)
    """
    def __init__(self, T=1.0, alpha=1.0):
        super(SoftCELoss, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, logits, targets, hard_targets=None):
        ce_loss = soft_cross_entropy(logits, targets, T=self.T)
        if hard_targets is not None and self.alpha != 0.0:
            ce_loss += self.alpha*F.cross_entropy(logits, hard_targets)
        return ce_loss

def soft_cross_entropy(logits, target, T=1.0, size_average=True, target_is_prob=False):
    """ Cross Entropy for soft targets
    
    **Parameters:**
        - **logits** (Tensor): logits score (e.g. outputs of fc layer)
        - **targets** (Tensor): logits of soft targets
        - **T** (float): temperature　of distill
        - **size_average**: average the outputs
        - **target_is_prob**: set True if target is already a probability.
    """
    if target_is_prob:
        p_target = target
    else:
        p_target = F.softmax(target/T, dim=1)
    
    logp_pred = F.log_softmax(logits/T, dim=1)
    # F.kl_div(logp_pred, p_target, reduction='batchmean')*T*T
    ce = torch.sum(-p_target * logp_pred, dim=1)
    if size_average:
        return ce.mean() * T * T
    else:
        return ce * T * T
