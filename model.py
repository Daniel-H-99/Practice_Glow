import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt

from utils import seq2sen

class RealNVP(nn.Module):
    def __init__(self, args):
        super(RealNVP, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        self.shape = (args.channels, args.height, args.width)   # shape of single input
        shape = self.shape
        
        for k in range(args.K):     # construct a sequence of coupling blocks of length K
            self.layers.append(Flowstep(args, shape))
            
        self.layers.append(Split(args, shape, is_last=True))    # does not conduct splitting, but exports latent vectors
    def forward(self, input, loss=0.0):
        return self.encode(input, loss)
    def encode(self, x, loss=0.0):
        # x: (batch x channel x height x width)
        assert x.shape[1:] == self.shape
        z = None
        for layer in self.layers:
            x, loss, z = layer(x, loss, z)
        return z, loss
    def decode(self, z):
        # z: (batch x pixels)
        x = None
        for layer in reversed(self.layers):
            z, x = layer.decode(z, x)
        assert x.shape[1:] == self.shape
        return x   # (batch x channels x height x width)
    
class Glow(nn.Module):
    def __init__(self, args):
        super(Glow, self).__init__()
        self.args = args
        self.layers = nn.ModuleList()
        self.shape = (args.channels, args.height, args.width)   # shape of single input
        shape = self.shape
        
        for l in range(args.L):     # construct a flow of size (L x K)
            self.layers.append(Squeeze(args))
            shape = (shape[0] * 4, shape[1] // 2, shape[2] // 2)
            for k in range(args.K):
                self.layers.append(Flowstep(args, shape))
            if l < args.L - 1:
                self.layers.append(Split(args, shape))
                shape = (shape[0] // 2, shape[1], shape[2])
            else:
                self.layers.append(Split(args, shape, is_last=True))
    def forward(self, input, loss=0.0):
        return self.encode(input, loss)
    def encode (self, x, loss):
        # x: (batch x channel x height x width)
        assert x.shape[1:] == self.shape
        z = None
        for layer in self.layers:
            x, loss, z = layer(x, loss, z)
        return z, loss
    def decode(self, z):
        # z: (batch x pixels)
        x = None
        for layer in reversed(self.layers):
            z, x = layer.decode(z, x)
        assert x.shape[1:] == self.shape
        return x   # (batch x channels x height x width)
    
class Squeeze(nn.Module):
    def __init__(self, args):
        super(Squeeze, self).__init__()
    def forward(self, x, loss, z):
        # (c x h x w) -> (c*4 x h/2 x w/2)
        b, c, h, w = x.shape
        x = x.view(b, c, h // 2, 2, w // 2, 2).permute(0, 1, 3, 5, 2, 4).contiguous().view(b, 4 * c, h // 2, w // 2)
        return x, loss, z
    def decode(self, z, x):
        # (c x h x w) -> (c*4 x h/2 x w/2)
        b, c, h, w = x.shape
        x = x.view(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).contiguous().view(b, c // 4, 2 * h, 2 * w)
        return z, x
    
class Flowstep(nn.Module):
    def __init__(self, args, shape):
        super(Flowstep, self).__init__()
        self.shape = shape
        self.layers = nn.ModuleList()   # (Actnorm + InvConv + AffineCoupling)
        self.layers.append(Actnorm(args, shape))
        self.layers.append(InvConv(args, shape))
        self.layers.append(AffineCoupling(args, shape))
    def forward(self, x, loss, z, decode=False):
        assert x.shape[1:] == self.shape
        for layer in self.layers:
            x, loss, z = layer(x, loss, z)
        return x, loss, z
    def decode(self, z, x):
        for layer in reversed(self.layers):
            z, x = layer.decode(z, x)
        return z, x
    
class Split(nn.Module):
    def __init__(self, args, shape, is_last=False):
        super(Split, self).__init__()
        self.shape = shape
        self.is_last = is_last
    def forward(self, x, loss, z):  # export half of latent vector and remaining to next step
        shape = x.shape
        assert shape[1:] == self.shape
        if not self.is_last:
            x, pre_z = x.chunk(2, dim=1)
        else:
            x, pre_z = None, x
        pre_z = pre_z.view(shape[0], -1)
        if z is None:
            z = pre_z
        else:
            z = torch.cat((z, pre_z), dim=1)
        return x, loss, z
    def decode(self, z, x):     # import latent vector and combine with back propagated representation from next step
        c, w, h = self.shape
        required_pixels = (c // 2) * w * h if not self.is_last else c * w * h
        z, pre_x = z[:, :-required_pixels], z[:, -required_pixels:]
        pre_x = pre_x.view(-1, required_pixels // (w * h), w, h)
        if not self.is_last:
            assert (x.shape[1:] == (c // 2, w, h))
            x = torch.cat((x, pre_x), dim=1)
        else:
            x = pre_x
        assert x.shape[1:] == self.shape
        return z, x
    
class Actnorm(nn.Module):
    def __init__(self, args, shape):
        super(Actnorm, self).__init__()
        self.shape = shape
        self.initialized = False
        self.bias = nn.Parameter(torch.zeros(1, shape[0], 1, 1))   # (1 x channels x 1 x 1)
        self.log_scale = nn.Parameter(torch.zeros(1, shape[0], 1, 1))    # (1 x channels x 1 x 1)
    def forward(self, x, loss, z):   # s * (x + b)
        shape = x.shape
        assert shape[1:] == self.shape
        if not self.initialized:
            x = x.transpose(0, 1).contiguous().view(shape[1], -1)   # (channels x -1)
            scale = 1 / (x.std(dim=1) + 1e-6)
            self.log_scale.data.copy_(scale.log().unsqueeze(0).unsqueeze(2).unsqueeze(3))
            self.bias.data.copy_(- x.mean(dim=1).unsqueeze(0).unsqueeze(2).unsqueeze(3))
            self.initialized = True
            x = x.view(shape[1], shape[0], shape[2], shape[3]).transpose(0, 1)
        x = self.log_scale.exp() * (x + self.bias)
        loss -= shape[2] * shape[3] * self.log_scale.sum()
        return x, loss, z
    def decode(self, z, x):   # x / s - b
        x = x / self.log_scale.exp() - self.bias
        assert x.shape[1:] == self.shape
        return z, x
    
class InvConv(nn.Module):
    def __init__(self, args, shape):
        super(InvConv, self).__init__()
        self.args = args
        self.shape = shape
        self.kernel = nn.Parameter(torch.randn(shape[0], shape[0]).qr()[0])
    def forward(self, x, loss, z):  # W x
        shape = x.shape
        assert shape[1:] == self.shape
        x = F.conv2d(x, self.kernel.unsqueeze(2).unsqueeze(3))
        loss -= shape[2] * shape[3] * torch.slogdet(self.kernel)[1]
        return x, loss, z
    def decode(self, z, x):     # W^(-1) x
        x = F.conv2d(x, self.kernel.double().inverse().float().unsqueeze(2).unsqueeze(3))
        assert x.shape[1:] == self.shape
        return z, x
    
class AffineCoupling(nn.Module):
    def __init__(self, args, shape):
        super(AffineCoupling, self).__init__()
        self.shape = shape
        self.nn = NN(args, (shape[0] // 2, shape[1], shape[2]))
        self.scale = nn.Parameter(torch.ones(shape[0] // 2, 1, 1))
    def forward(self, x, loss, z):  # cat(x1, s * (x2 + t))
        shape = x.shape
        assert shape[1:] == self.shape
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.nn(x1)
        # s = F.sigmoid(s + 2.0)
        s = self.scale * torch.tanh(s)
        x2 = s.exp() * (x2 + t)
        x = torch.cat((x1, x2), dim=1)
        loss -= s.view(shape[0], -1).sum(dim=1).mean(dim=0)
        return x, loss, z
    def decode(self, z, x):     # cat(x1, x2 / s - t))
        x1, x2 = x.chunk(2, dim=1)
        s, t = self.nn(x1)
        # s = F.sigmoid(s + 2.0)
        s = self.scale * torch.tanh(s)
        x2 = x2 / s.exp() - t
        x = torch.cat((x1, x2), dim=1)
        assert x.shape[1:] == self.shape
        return z, x

class NN(nn.Module):
    def __init__(self, args, shape):    # (3 conv + 2 ReLU)
        super(NN, self).__init__()
        self.shape = shape
        self.layers = nn.Sequential(nn.Conv2d(shape[0], args.nn_channels, args.nn_kernel, stride=1, padding=(args.nn_kernel - 1) // 2),
                                     nn.ReLU(),
                                     nn.Conv2d(args.nn_channels, args.nn_channels, 1, stride=1),
                                     nn.ReLU(),
                                     nn.Conv2d(args.nn_channels, shape[0] * 2, args.nn_kernel, stride=1, padding=(args.nn_kernel - 1) // 2))
        nn.init.zeros_(self.layers[4].weight)
        nn.init.zeros_(self.layers[4].bias)
    def forward(self, x):   # x -> s, t
        assert x.shape[1:] == self.shape
        x = self.layers(x)
        s, t = x.chunk(2, dim=1)
        return s, t
    
        
            