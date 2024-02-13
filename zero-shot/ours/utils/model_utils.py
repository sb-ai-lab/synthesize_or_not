import numpy as np
import pandas as pd
import os 
from tqdm import tqdm
import numba as nb
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
import joblib
import scipy.special
from datetime import datetime
from torch import Tensor
from typing import Union, Tuple, List, Dict
import random
from copy import deepcopy
import matplotlib.pyplot as plt
import transformers
from transformers import get_scheduler, get_cosine_schedule_with_warmup
from utils.plot_utils import plot_loss, plot_lr, plot_series

def lengths_to_mask(lengths, max_len=None, dtype=None):
    """
    Converts a "lengths" tensor to its binary mask representation.
    
    Based on: https://discuss.pytorch.org/t/how-to-generate-variable-length-mask/23397
    
    :lengths: N-dimensional tensor
    :returns: N*max_len dimensional tensor. If max_len==None, max_len=max(lengtsh)
    """
    assert len(lengths.shape) == 1, 'Length shape should be 1 dimensional.'
    max_len = max_len or lengths.max().item()
    mask = torch.arange(
        max_len,
        device=lengths.device,
        dtype=lengths.dtype)\
    .expand(len(lengths), max_len) < lengths.unsqueeze(1)
    if dtype is not None:
        mask = torch.as_tensor(mask, dtype=dtype, device=lengths.device)
    return mask

class SequenceAbstractPooler(nn.Module):
    """Abstract pooling class."""

    def __init__(self):
        super(SequenceAbstractPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """Forward-call."""
        return self.forward(*args, **kwargs)
    
class SequenceMaxPooler(SequenceAbstractPooler):
    """Max value pooling."""

    def __init__(self):
        super(SequenceMaxPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, -float("inf"))
        values, _ = torch.max(x, dim=-2)
        return values
    
class SequenceAvgPooler(SequenceAbstractPooler):
    """Mean value pooling."""

    def __init__(self):
        super(SequenceAvgPooler, self).__init__()

    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        """Forward-pass."""
        x = x.masked_fill(~x_mask, 0)
        x_active = torch.sum(x_mask, dim=-2)
        x_active = x_active.masked_fill(x_active == 0, 1)
        values = torch.sum(x, dim=-2) / x_active.data
        return values
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.0, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)#.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe
        return x
    
class NLinearMemoryEfficient(nn.Module):
    """Linear multi-dim embedding from https://github.com/yandex-research/tabular-dl-num-embeddings/tree/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8.

    Args:
        n : num of features.
        d_in: input size.
        d_out: output size.
    """

    def __init__(self, n: int, d_in: int, d_out: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for _ in range(n)])

    def forward(self, x):
        """Forward-pass."""
        return torch.stack([l(x[:, i]) for i, l in enumerate(self.layers)], 1)


class Periodic(nn.Module):
    """Periodic positional embedding for numeric features from https://github.com/yandex-research/tabular-dl-num-embeddings/tree/c1d9eb63c0685b51d7e1bc081cdce6ffdb8886a8.

    Args:
        n_features: num of numeric features
        emb_size: output size will be 2*emb_size
        sigma: weights will be initialized with N(0,sigma)
        flatten_output: if flatten output or not.
    """

    def __init__(
        self, n_features: int, emb_size: int = 64, sigma: float = 0.05, flatten_output: bool = False, **kwargs
    ) -> None:
        super().__init__()
        self.n_features = n_features
        self.emb_size = emb_size
        coefficients = torch.normal(0.0, sigma, (n_features, emb_size))
        self.coefficients = nn.Parameter(coefficients)
        self.flatten_output = flatten_output

    @staticmethod
    def _cos_sin(x: Tensor) -> Tensor:
        return torch.cat([torch.cos(x), torch.sin(x)], -1)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.emb_size * 2 * self.n_features
        else:
            return self.n_features

    def forward(self, x: Tensor) -> Tensor:
        """Forward-pass."""
        x = self._cos_sin(2 * np.pi * self.coefficients[None] * x[..., None])
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x
    
class PLREmbedding(nn.Module):
    """ReLU ◦ Linear ◦ Periodic embedding for numeric features from https://arxiv.org/pdf/2203.05556.pdf.

    Args:
        num_dims: int
        emb_size: int
        sigma: float
        flatten_output : bool
    """

    def __init__(
        self,
        num_dims: int,
        embedding_size: Union[int, Tuple[int, ...], List[int]] = 64,
        emb_size_periodic: int = 64,
        sigma_periodic: float = 0.05,
        flatten_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_dims = num_dims
        self.embedding_size = embedding_size
        self.layers: list[nn.Module] = []
        self.layers.append(Periodic(num_dims, emb_size_periodic, sigma_periodic))
        self.layers.append(NLinearMemoryEfficient(num_dims, 2 * emb_size_periodic, embedding_size))
        self.layers.append(nn.ReLU())
        self.layers = nn.Sequential(*self.layers)
        self.flatten_output = flatten_output

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape.

        """
        if self.flatten_output:
            return self.num_dims * self.embedding_size
        else:
            return self.num_dims

    def forward(self, X: Dict) -> Tensor:
        """Produce embedding for each value in input.

        Args:
            X : Dict

        Returns:
            torch.Tensor

        """
        x = self.layers(X)
        if self.flatten_output:
            return x.view(x.shape[0], -1)
        return x
    
class MaskedBatchNorm1d(nn.BatchNorm1d):
    """
    Masked verstion of the 1D Batch normalization.
    
    Based on: https://github.com/ptrblck/pytorch_misc/blob/20e8ea93bd458b88f921a87e2d4001a4eb753a02/batch_norm_manual.py
    
    Receives a N-dim tensor of sequence lengths per batch element
    along with the regular input for masking.
    
    Check pytorch's BatchNorm1d implementation for argument details.
    """
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(MaskedBatchNorm1d, self).__init__(
            num_features,
            eps,
            momentum,
            affine,
            track_running_stats
        )

    def forward(self, inp, lengths):
        self._check_input_dim(inp)

        exponential_average_factor = 0.0
        
        # We transform the mask into a sort of P(inp) with equal probabilities
        # for all unmasked elements of the tensor, and 0 probability for masked
        # ones.
        mask = lengths_to_mask(lengths, max_len=inp.shape[-1], dtype=inp.dtype)
        n = mask.sum()
        mask = mask / n
        mask = mask.unsqueeze(1).expand(inp.shape)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training and n > 1:
            # Here lies the trick. Using Var(X) = E[X^2] - E[X]^2 as the biased
            # variance, we do not need to make any tensor shape manipulation.
            # mean = E[X] is simply the sum-product of our "probability" mask with the input...
            mean = (mask * inp).sum([0, 2])
            # ...whereas Var(X) is directly derived from the above formulae
            # This should be numerically equivalent to the biased sample variance
            var = (mask * inp ** 2).sum([0, 2]) - mean ** 2
            with torch.no_grad():
                self.running_mean = exponential_average_factor * mean\
                    + (1 - exponential_average_factor) * self.running_mean
                # Update running_var with unbiased var
                self.running_var = exponential_average_factor * var * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_var
        else:
            mean = self.running_mean
            var = self.running_var

        inp = (inp - mean[None, :, None]) / (torch.sqrt(var[None, :, None] + self.eps))
        if self.affine:
            inp = inp * self.weight[None, :, None] + self.bias[None, :, None]

        return inp
    
class MaskedMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0
        flat_mask = torch.flatten(mask)
        assert(len(flat_mask) == len(diff2))
        sum2 = torch.sum(diff2 * flat_mask) / torch.count_nonzero(flat_mask)

        return sum2
    
class MaskedRMSELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedRMSELoss, self).__init__()

    def forward(self, input, target, mask):
        diff2 = (torch.flatten(input) - torch.flatten(target)) ** 2.0
        flat_mask = torch.flatten(mask)
        assert(len(flat_mask) == len(diff2))
        sum2 = torch.sqrt(torch.sum(diff2 * flat_mask) / torch.count_nonzero(flat_mask))

        return sum2

class MaskedSMAPELoss(torch.nn.Module):
    def __init__(self):
        super(MaskedSMAPELoss, self).__init__()

    def forward(self, input, target, mask):
        target_min = torch.transpose(torch.min(target, dim=1).values.expand(target.shape[1], target.shape[0]), 0, 1)
        flat_input = torch.flatten(input + target_min + 1)
        flat_target = torch.flatten(target + target_min + 1)
        flat_mask = torch.flatten(mask)
        masked_target = flat_mask * flat_target
        masked_input = flat_mask * flat_input
        abs = torch.abs(masked_input - masked_target)
        abs_target = torch.abs(masked_target)
        abs_input = torch.abs(masked_input)
        loss = torch.sum(torch.div(abs, abs_target + abs_input + 1e-8)) / torch.count_nonzero(flat_mask)
        return loss
    
class ICTransformer(nn.Module):
    def __init__(self, input_size=1,
                 hidden_size=128,
                 num_layers=1,
                 bidirectional=True
                 ):
        
        super().__init__()
        self.input_size = input_size
        # self.input_size = input_size - 1 + 3 + 16
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # main transformer
        tr = nn.TransformerEncoderLayer(d_model=self.hidden_size, dim_feedforward=self.hidden_size * 4, nhead=4, batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(tr, num_layers=5)

        # projection layer: (bs, sec, feat) -> (bs, sec, hidden)
        self.proj = nn.Linear(1, self.hidden_size)

        # last linear layer + init bias
        self.fc = nn.Linear(self.hidden_size, 720)
        # bias = torch.Tensor((4.299640889483169e-06, 0.0001597132243785138, 0.030531974279254418))
        # self.fc.bias.data = bias
        # shape = self.fc.weight.data.shape
        # self.fc.weight.data = torch.zeros(shape[0], shape[1], requires_grad=True)

        # bns
        self.out_bn = nn.BatchNorm1d(num_features=self.hidden_size)
        #self.inp_bn = MaskedBatchNorm1d(num_features=self.input_size) # masked bn
        self.inp_bn = nn.BatchNorm1d(num_features=1) # masked bn

        # embeddings + pe
        self.cls = nn.Embedding(num_embeddings=2, embedding_dim=self.hidden_size)
        self.embedder = PLREmbedding(num_dims=self.input_size, embedding_size=self.hidden_size, emb_size_periodic=500)
        # self.coord_embedder = nn.Embedding.from_pretrained(torch.Tensor(coords_np), freeze=True)
        # self.sensor_embedder = nn.Embedding(num_embeddings=5200, embedding_dim=16)

        pe = torch.zeros(500+1, self.hidden_size)
        position = torch.arange(0, 500+1, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.hidden_size, 2).float() * (-math.log(10000.0) / self.hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #.transpose(0, 1)
        self.pe = nn.Parameter(pe)
        
        # pe = PositionalEncoding(d_model=self.hidden_size, max_len=500+1).get_pe()
        # self.pe = nn.Parameter(pe)
        
        # self.pe = nn.Parameter(data=torch.randn(500+1, self.hidden_size))
        # self.layernorm = nn.LayerNorm(self.hidden_size)
        # self.dropout = nn.Dropout(p = 0.2)
        # nn.init.constant_(self.pe.weight, pe)

        # poolings
        self.max = SequenceMaxPooler()
        self.mean = SequenceAvgPooler()

        # act
        self.act = nn.LeakyReLU()
        self.cls2 = nn.Embedding(num_embeddings=2, embedding_dim=20)

        
    def forward(self, x, mask):

        # create embeddings from sensor_id
        # embeds = self.coord_embedder(x[:, :, 0].long()) 
        # embeds2 = self.sensor_embedder(x[:, :, 0].long()) 
        #x = self.embedder(x)
        #print('x', x.shape)
        #print('mask', mask.sum(dim=-1))

        x = torch.unsqueeze(x, dim=-1)
        #x = torch.permute(x, (0, 2, 1))
        #x = self.inp_bn(x)
        #x = torch.permute(x, (0, 2, 1))
        proj = self.proj(x)
        # prepare padding mask
        x_len = mask.sum(dim=-1)
        # mask = ~x[:, :, -1].bool()
        mask = ~mask.bool()

        # add one more element in the mask for classification token
        mask = torch.cat((torch.zeros((x.shape[0], 1)).bool().to(x.device), mask), dim = 1)

        # concat embeds with features
        #x = torch.cat([x[:, :, 1:], embeds], dim=-1) # ????

        # (bs, sec, feat) -> (bs, feat, sec) -> batch norm ->(bs, sec, feat)
        # print(x.shape)
        #print('x', x.shape)
        #x = torch.permute(x, (0, 2, 1))
        #x = self.inp_bn(x, x_len)
        
        # project input features to the hidden dim
        # x = self.proj(x)
        #print('x', x.shape)
        #x = torch.permute(x, (0, 2, 1))
        #print('new', proj.shape)
        # add classification token
        x_cls = self.cls(torch.zeros((x.shape[0], 1)).long().to(x.device))
        x = torch.cat((x_cls, proj), dim = 1)
        #print('after cls', x.shape)
        # possible positional encoding
        # x = self.pe(x)
        x = x + self.pe
        # x = self.layernorm(x)
        # x = self.dropout(x)
        #print('after pe', x.shape)
        # transformer model
        x = self.tr(x, 
                    src_key_padding_mask=mask
                    ) # 
        #print('after tr', x.shape)
        # create mask for calculating poolings
        mask = (torch.arange(x.shape[1])[None, :].to(x.device) < x_len[:, None]+1)[:, :, None]

        # max and mean poolings
        # feature_max = self.max(x, mask)
        # feature_avg = self.mean(x, mask)

        # concat cls pooling, emb for the first element, + 2 poolings
        # out = torch.cat((x[:, 0, :], feature_max, feature_avg), dim = 1)
        # x[:,1,:] - выше заменить на последний ненулевой по маске (можно из длинны + 1 взять)
        # MLP
        #out = self.out_bn(x[:, 0, :])
        out = x[:, 0, :]
        out = self.act(out)
        out = self.fc(out)
        #1/0
        #out = self.cls2(torch.ones((x.shape[0], 1)).long().to(x.device)).squeeze(1)
        return out
    
# def train_one_epoch(model, loss_fn, optimizer, scheduler, train_loader, device):
#     lr_list = []
#     loss_list = []

#     for i, data in enumerate(train_loader):
#         series = data['series'].to(device)
#         mask_series = data['mask_series'].to(device)
#         target = data['target'].to(device)
#         mask_target = data['mask_target'].to(device)
#         optimizer.zero_grad()
#         outputs = model(series, mask_series)
#         targets = target * mask_target
#         outputs = outputs * mask_target
#         loss = loss_fn(outputs, targets, mask_target) # / torch.sum(data['mask_target'].to(device)) * data['target'].to(device).shape[1] * data['target'].to(device).shape[0]
#         lr_list.append(scheduler.get_lr())

#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         loss_list.append(loss.item())
#         print(loss)

#     return loss_list, lr_list
    

    


