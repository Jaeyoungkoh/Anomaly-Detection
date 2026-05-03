import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.attn_dcdetector import DAC_structure, AttentionLayer
from layers.attn_dcdetector import DataEmbedding, TokenEmbedding
from layers.RevIN import RevIN
from tkinter import _flatten

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=None):
        series_list = []
        prior_list = []
        for attn_layer in self.attn_layers:
            series, prior = attn_layer(x_patch_size, x_patch_num, x_ori, patch_index, attn_mask=attn_mask)
            series_list.append(series)
            prior_list.append(prior)
        return series_list, prior_list


class DCdetector(nn.Module):
    def __init__(self, args):
        super(DCdetector, self).__init__()

        self.win_size = args.win_size
        self.enc_in = args.input_c
        self.c_out = args.input_c
        self.n_heads = args.n_heads_dc
        self.d_model = args.d_model_dc
        self.e_layers = args.e_layers_dc
        self.patch_size = args.patch_size
        self.channel = args.input_c
        self.d_ff = 512
        self.dropout = 0.0
        self.activation = 'gelu'
        self.output_attention = True

        self.revin_layer = RevIN(num_features=args.input_c)

        # Patching List  
        self.embedding_patch_size = nn.ModuleList()
        self.embedding_patch_num = nn.ModuleList()
        for i, patchsize in enumerate(self.patch_size):
            self.embedding_patch_size.append(DataEmbedding(patchsize, self.d_model, self.dropout))
            self.embedding_patch_num.append(DataEmbedding(self.win_size//patchsize, self.d_model, self.dropout))

        self.embedding_window_size = DataEmbedding(self.enc_in, self.d_model, self.dropout)
        
        # Dual Attention Encoder
        self.encoder = Encoder(
            [
                AttentionLayer(
                    DAC_structure(self.win_size, self.patch_size, self.channel, False, attention_dropout=self.dropout, output_attention=self.output_attention),
                    self.d_model, self.patch_size, self.channel, self.n_heads, self.win_size)for l in range(self.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(self.d_model)
        )

        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)


    def forward(self, x):
        B, L, M = x.shape #Batch win_size channel
        series_patch_mean = []
        prior_patch_mean = []

        # Instance Normalization Operation
        x = self.revin_layer(x, 'norm')
        x_ori = self.embedding_window_size(x)
        
        # Mutil-scale Patching Operation 
        for patch_index, patchsize in enumerate(self.patch_size):
            x_patch_size, x_patch_num = x, x
            x_patch_size = rearrange(x_patch_size, 'b l m -> b m l') #Batch channel win_size
            x_patch_num = rearrange(x_patch_num, 'b l m -> b m l') #Batch channel win_size
            
            x_patch_size = rearrange(x_patch_size, 'b m (n p) -> (b m) n p', p = patchsize) 
            x_patch_size = self.embedding_patch_size[patch_index](x_patch_size)
            x_patch_num = rearrange(x_patch_num, 'b m (p n) -> (b m) p n', p = patchsize) 
            x_patch_num = self.embedding_patch_num[patch_index](x_patch_num)
            
            series, prior = self.encoder(x_patch_size, x_patch_num, x_ori, patch_index)
            series_patch_mean.append(series), prior_patch_mean.append(prior)

        series_patch_mean = list(_flatten(series_patch_mean))
        prior_patch_mean = list(_flatten(prior_patch_mean))
            
        if self.output_attention:
            return series_patch_mean, prior_patch_mean
        else:
            return None
        