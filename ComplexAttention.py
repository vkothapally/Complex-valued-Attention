import torch
import torch.nn as nn
import numpy as np
from ComplexFunctions import *


def param(nnet, Mb=True): return np.round(sum([param.nelement() for param in nnet.parameters()]) / 10**6 if Mb else neles,2)

class ComplexScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product for Complex Numbers '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None):
        attn = Complexbmm(q / self.temperature, k.conj().transpose(-1,-2)).abs()
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        attn     = self.dropout(self.softmax(attn))
        output   = torch.view_as_complex(torch.stack([torch.matmul(attn,v.real), torch.matmul(attn,v.imag)],-1))
        # output   = Complexbmm(attn, v)
        return output, attn


class ComplexMultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module for Complex Numbers '''

    def __init__(self, n_head, f_in, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = f_in // n_head
        self.d_v = f_in // n_head

        self.w_qs = ComplexLinear(f_in, n_head * self.d_k, bias=False)
        self.w_ks = ComplexLinear(f_in, n_head * self.d_k, bias=False)
        self.w_vs = ComplexLinear(f_in, n_head * self.d_v, bias=False)
        self.fc   = ComplexLinear(n_head * self.d_v, f_in, bias=False)

        self.attention  = ComplexScaledDotProductAttention(temperature=self.d_k ** 0.5)
        self.dropout    = ComplexDropout(dropout)
        self.layer_norm = ComplexLayerNorm(f_in, epsilon=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ComplexPositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module for Complex Numbers '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = ComplexLinear(d_in, d_hid) # position-wise
        self.w_2 = ComplexLinear(d_hid, d_in) # position-wise
        self.l_norm  = ComplexLayerNorm(d_in, epsilon=1e-6)
        self.dropout = ComplexDropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(self.w_1(x))
        x = self.dropout(x)
        x += residual

        x = self.l_norm(x)

        return x


class ComplexSelfAttention(nn.Module):
    ''' Input: [Batch x Time x Features]  Complex Self-Attention for Complex Numbers '''

    def __init__(self, f_in, f_out, n_head, dropout=0.1):
        super(ComplexSelfAttention, self).__init__()
        self.slf_attn = ComplexMultiHeadAttention(n_head, f_in, dropout=dropout)
        self.pos_ffn  = ComplexPositionwiseFeedForward(f_in, f_out, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn



if __name__=='__main__':
    cplxmix     = torch.view_as_complex(torch.randn(10,400,64,2)).to('cuda')
    cplxmodel   = ComplexSelfAttention(f_in=64, f_out=16, n_head=4, dropout=0.1).to('cuda')
    cplxmix_attn, cplxmix_slf_attn = cplxmodel(cplxmix)
    print('\n\n--------------------------------- Script Inputs and Outputs :: Summary')
    print('Model params (M) : ', param(cplxmodel), 'M   ---> Complex Version')
    print('Input Mix audio  : ', cplxmix.real.shape, cplxmix.imag.shape)
    print('Output Attn      : ', cplxmix_attn.real.shape, cplxmix_attn.imag.shape)
    print('Output Self Attn : ', cplxmix_slf_attn.shape)
    print('--------------------------------------------------------------------------\n')
    print('Done!')
