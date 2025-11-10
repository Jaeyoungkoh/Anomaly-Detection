import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from utils.dlutils import *
from utils.constants import *

class TranAD(nn.Module):
	def __init__(self, args):
		
		super(TranAD, self).__init__()
		
		self.lr = args.lr
		self.batch = args.batch_size
		self.n_feats = args.input_c
		self.n_window = args.win_size
		self.n = self.n_feats * self.n_window
		
		self.pos_encoder = PositionalEncoding(2 * args.input_c, 0.1, self.n_window)
		encoder_layers = TransformerEncoderLayer(d_model=2 * args.input_c, nhead=args.input_c, dim_feedforward=64, dropout=0.1)
		self.transformer_encoder = TransformerEncoder(encoder_layers, 1)
		decoder_layers1 = TransformerDecoderLayer(d_model=2 * args.input_c, nhead=args.input_c, dim_feedforward=64, dropout=0.1)
		self.transformer_decoder1 = TransformerDecoder(decoder_layers1, 1)
		decoder_layers2 = TransformerDecoderLayer(d_model=2 * args.input_c, nhead=args.input_c, dim_feedforward=64, dropout=0.1)
		self.transformer_decoder2 = TransformerDecoder(decoder_layers2, 1)
		self.fcn = nn.Sequential(nn.Linear(2 * args.input_c, args.input_c), nn.Sigmoid())

	def encode(self, src, c, tgt):
		src = torch.cat((src, c), dim=2)
		src = src * math.sqrt(self.n_feats)
		src = self.pos_encoder(src)
		memory = self.transformer_encoder(src, mask=None)
		tgt = tgt.repeat(1, 1, 2)
		return tgt, memory

	def forward(self, src):
		# src : (window_size, batch_size, n_features)
		tgt = src
		
		# Phase 1 - Without anomaly scores
		c = torch.zeros_like(src)
		# x1 shape: (window_size, batch_size, n_features)
		x1 = self.fcn(self.transformer_decoder1(*self.encode(src, c, tgt)))
		
		# Phase 2 - With anomaly scores
		c = (x1 - src) ** 2
		# x2 shape: (window_size, batch_size, n_features)
		x2 = self.fcn(self.transformer_decoder2(*self.encode(src, c, tgt)))
		
		return x1, x2