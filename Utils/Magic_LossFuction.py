import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneralizeCrossEntropy(nn.Module):
    def __init__(self, esp=1e-5):
        super(GeneralizeCrossEntropy, self).__init__()
        self.esp = esp
    
    def cul_q(self, snrs):
        q = 1 - (torch.mean(snrs.to(torch.float)) + 20) / 38
        return q

    def forward(self, logits, labels, snrs):
        # Negative box cox: (1-f(x)^q)/q
        q = self.cul_q(snrs)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        probs = F.softmax(logits, dim=-1)
        loss = (1 - torch.pow(torch.sum(labels * probs, dim=-1), q)) / (q + self.esp)
        loss = torch.mean(loss)
        return loss
    
class NoiseRobustCrossEntropy(nn.Module):
    def __init__(self, mid_snr=-2, a=0.01, reduction='mean'):
        super(NoiseRobustCrossEntropy, self).__init__()
        self.mid_snr = mid_snr
        self.a = a
        self.reduction = reduction
        
    def cul_k(self, snrs):
        snrs = self.a * (snrs - self.mid_snr)
        k = torch.exp(snrs)
        k = k.reshape(-1, 1)
        return k
        
    def forward(self, logits, labels, snrs):
        k = self.cul_k(snrs)
        labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[-1])
        logits = F.softmax(logits, dim=-1)
        loss = -k * logits.log() * labels
        loss = torch.sum(loss, dim=1)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss
        