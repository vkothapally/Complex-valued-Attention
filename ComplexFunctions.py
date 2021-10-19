import torch
import torch.nn as nn


def Complexbmm(x, y):
    ''' Batch Matrix Multiplication for Complex Numbers '''
    real = torch.matmul(x.real, y.real) - torch.matmul(x.imag, y.imag)
    imag = torch.matmul(x.real, y.imag) + torch.matmul(x.imag, y.real)
    out  = torch.view_as_complex(torch.stack([real, imag], -1))
    return out

class ComplexLinear(nn.Module):
    ''' [nn.Linear] Fully Connected Layer for Complex Numbers '''
    def __init__(self, in_features, out_features, bias=True):
        super(ComplexLinear, self).__init__()
        self.real_linear = nn.Linear(in_features, out_features, bias=bias)
        self.imag_linear = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x):
        real = self.real_linear(x.real) - self.imag_linear(x.imag)
        imag = self.real_linear(x.imag) + self.imag_linear(x.real)
        out  = torch.view_as_complex(torch.stack([real, imag], -1))
        return out


class ComplexDropout(nn.Module):
    ''' [nn.Dropout] DropOut for Complex Numbers '''
    def __init__(self, p=0.1, inplace=False):
        super(ComplexDropout, self).__init__()
        self.drop = nn.Dropout(p=p, inplace=inplace)

    def forward(self, x):
        x.imag = x.imag + 1e-10
        mag, phase = self.drop(x.abs()), x.angle()
        real, imag = mag * torch.cos(phase), mag * torch.sin(phase)
        out  = torch.view_as_complex(torch.stack([real, imag], -1))
        return out

class ComplexSoftMax(nn.Module):
    ''' [nn.Softmax] SoftMax for Complex Numbers '''
    def __init__(self, dim=-1):
        super(ComplexSoftMax, self).__init__()
        self.softmax = nn.Softmax(dim=dim)

    def forward(self, x):
        x.imag = x.imag + 1e-10
        mag, phase = self.softmax(x.abs()), x.angle()
        real, imag = mag * torch.cos(phase), mag * torch.sin(phase)
        out  = torch.view_as_complex(torch.stack([real, imag], -1))
        return out

class ComplexLayerNorm(nn.Module):
    ''' [nn.LayerNorm] LayerNorm for Complex Numbers '''
    def __init__(self, normal_shape, affine=True, epsilon=1e-10):
        super(ComplexLayerNorm, self).__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        self.affine  = affine 
        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
            self.beta  = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            self.gamma.data.fill_(1)
            self.beta.data.zero_()

    def forward(self, x):
        dim  = list(range(1,len(x.shape)))
        mean = torch.view_as_complex(torch.stack((x.real.mean(dim=dim, keepdim=True), x.imag.mean(dim=dim, keepdim=True)),-1))
        x_mean = (x - mean)
        std  = ((x_mean * x_mean.conj()).abs() + self.epsilon).sqrt()
        y    = torch.view_as_complex(torch.stack((x_mean.real/std, x_mean.imag/std),-1))
        if self.affine:
            y = (self.gamma * y) + self.beta
        return y








