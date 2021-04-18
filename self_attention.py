from transformers import get_cosine_schedule_with_warmup 
from transformers import get_linear_schedule_with_warmup 
from transformers import get_constant_schedule
from transformers import AdamW

from torch import nn
from torch.nn.utils.rnn import pad_sequence as pad 
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch

from hyper_params import *
import math
import torch.nn.functional as F

class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.init_weights()
        self.relu = nn.ReLU()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask,len_label):
      if len_label is None:
        return self.reasoning(src,src_mask)
      else:
        return self.forward2(src,src_mask,len_label)

    def forward2(self, src, src_mask,len_label):
        src = src.transpose(0,1)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        #residual connection
        output = src + output
        output = self.relu(output)

        output = self.decoder(output)
        output = output.transpose(0,1)
        s = output.size()
        if len_label <= s[1]:
          ret = output[:,:len_label]
        if len_label > s[1]:
          ret = torch.zeros(s[0],len_label,s[2])
          if USE_GPU:
            ret = ret.to("cuda")
          ret[:,:s[1]] = output
        return ret

    def reasoning(self, src, src_mask):
        src = src.transpose(0,1)
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        #residual connection
        #output = src + output
        #output = self.relu(output)
        output = self.decoder(output)
        output = output.transpose(0,1)
        return output.argmax(2)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def create_transformer(vocab_size,hidden):
  m = TransformerModel(vocab_size,hidden,8,hidden,2,DROPOUT)
  return m
