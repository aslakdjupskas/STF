import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = F.mse_loss(a, b).item()
    return -10 * math.log10(mse)

import torch

def rmse_and_snr(a: torch.Tensor, b: torch.Tensor) :
    se = (a - b) ** 2
    mse = torch.mean(se)
    rmse = torch.sqrt(mse)
    sp = torch.mean(a ** 2)
    ms_snr = sp / (mse + 1e-10)
    return rmse.detach(), ms_snr.detach()

