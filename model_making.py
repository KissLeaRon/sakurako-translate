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
from self_attention import create_transformer

###
#optimizer and scheduler
def optimizer_scheduler(model,batch_len):
  optimizer = AdamW(model.parameters(),lr = LEARNING_RATE, eps = 1e-8)
  #scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 5, num_training_steps = EPOCH)
  scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps =WARMING_UP*batch_len, num_training_steps = EPOCH*batch_len)  
  #scheduler = get_constant_schedule(optimizer)
  return (optimizer, scheduler)

def load_model(dict_size):
  vocab_size = dict_size+1
  hidden = 64
  #encoder = Encoder(vocab_size, hidden)
  #decoder = Decoder(vocab_size, hidden)
  #seq2seq = Seq2Seq(encoder,decoder)
  #seq2seq = VanillaLSTM(vocab_size,hidden)
  seq2seq = create_transformer(vocab_size,hidden)
  return seq2seq

class Encoder(nn.Module):
  def __init__(self, vocab_size, hidden_dim):
    super(Encoder, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.emb = nn.Embedding(vocab_size, hidden_dim,padding_idx=0)
    self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

  def forward(self,seq,length):
    emb = self.emb(seq)
    packed = pack(emb,length,batch_first=True,enforce_sorted=False)    
    out,(h,c) = self.lstm(packed)
    return out,h,c

class Decoder(nn.Module):
  def __init__(self, vocab_size,hidden_dim):
    super(Decoder, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.emb = nn.Embedding(vocab_size, hidden_dim,padding_idx=0)
    self.lstm = nn.LSTM(hidden_dim,hidden_dim, batch_first=True)
    self.lin = nn.Linear(hidden_dim, vocab_size)

  def forward(self,inp_one_len,h,c):
    inp = inp_one_len.unsqueeze(0).transpose(0,1)
    emb = self.emb(inp)
    out, (h,c) = self.lstm(emb,(h,c))
    pred = self.lin(out.squeeze(1))
    return pred, h, c

class Seq2Seq(nn.Module):
  def __init__(self,encoder,decoder,training=True):
    super().__init__()
    self.enc = encoder
    self.dec = decoder
    self.training = training
  def train(self):
    super().train()
    self.training = True
  def eval(self):
    super().train(False)
    self.training = False

  def forward(self,batch,lengths=None):
    if self.training:
      return self.forward_training(batch)
    if isinstance(batch,dict):
      return self.forward_training(batch,teacher_ratio=0)
    else:
      return self.reasoning(batch,lengths)

  def forward_training(self,batch,teacher_ratio = 0.5):
    old = batch["old"]
    len_old = batch["len_old"]
    new = batch["new"]
    len_new = batch["len_new"]
    
    if USE_GPU:
      old = old.to("cuda")
      new = new.to("cuda")

    len_predict = max(len_old)
    batch_size  = old.size()[0]
    ret = torch.zeros(batch_size,len_predict,self.dec.vocab_size)
    if USE_GPU:
      ret = ret.to("cuda")
    hs, hidden, cell = self.enc(new,len_new)

    inp_one_len = old[:,0]
    for l in range(1,len_predict):
      out, hidden, cell = self.dec(inp_one_len,hidden,cell)
      ret[:,l] = out
      force = torch.rand(1) < teacher_ratio
      top1id = out.argmax(1)
      inp_one_len = old[:,l] if teacher_ratio else top1id
    ret = ret[:,1:]
    return ret

  def reasoning(self,id_array,lengths):
    if USE_GPU:
      id_array = id_array.to("cuda")
    len_predict = 120
    ret = torch.zeros(1,len_predict,self.dec.vocab_size) 

    if USE_GPU:
      ret = ret.to("cuda")
    hs, hidden, cell = self.enc(id_array,lengths)

    inp_one_len = id_array[:,0]
    for l in range(1,len_predict):
      out, hidden, cell = self.dec(inp_one_len,hidden,cell)
      ret[:,l] = out
      top1id = out.argmax(1)
      inp_one_len = top1id
    return ret.argmax(2)

class VanillaLSTM(nn.Module):
  def __init__(self, vocab_size,hidden_dim):
    super(VanillaLSTM, self).__init__()
    self.vocab_size = vocab_size
    self.hidden_dim = hidden_dim
    self.emb = nn.Embedding(vocab_size, hidden_dim,padding_idx=0)
    self.lstm = nn.LSTM(hidden_dim,hidden_dim, batch_first=True)
    self.lin = nn.Linear(hidden_dim, vocab_size)

  def forward(self,batch,lengths=None):
    if self.training:
      return self.forward_training(batch)
    if isinstance(batch,dict):
      return self.forward_training(batch,teacher_ratio=0)
    else:
      return self.reasoning(batch,lengths)

  def forward_training(self,batch,teacher_ratio = 0.5):
    old = batch["old"]
    len_old = batch["len_old"]
    new = batch["new"]
    len_new = batch["len_new"]
    
    if USE_GPU:
      old = old.to("cuda")
      new = new.to("cuda")

    len_predict = max(len_old)
    batch_size  = old.size()[0]
    ret = torch.zeros(batch_size,len_predict,self.vocab_size)
    if USE_GPU:
      ret = ret.to("cuda")

    inp_one_len = old[:,0]
    inp = inp_one_len.unsqueeze(0).transpose(0,1)
    emb = self.emb(inp)
    out, (hidden, cell) = self.lstm(emb)
    out = self.lin(out.squeeze(1))
    ret[:,1] = out
    top1id = out.argmax(1)
    force = torch.rand(1) < teacher_ratio
    inp_one_len = old[:,1] if teacher_ratio else top1id

#old, _,a,b,c,...
#new, _,A,B,C,...

    for l in range(2,len_predict):
      inp = inp_one_len.unsqueeze(0).transpose(0,1)
      emb = self.emb(inp)
      out, (hidden, cell) = self.lstm(emb,(hidden,cell))
      out = self.lin(out.squeeze(1))
      ret[:,l] = out
      top1id = out.argmax(1)
      force = torch.rand(1) < teacher_ratio
      try:
        pos = old[:,l]
      except IndexError:
        pos = torch.zeros(inp_one_len.shape).long()
      inp_one_len = pos if teacher_ratio else top1id
    ret = ret[:,1:]
    return ret

  def reasoning(self,id_array,lengths):
    if USE_GPU:
      id_array = id_array.to("cuda")
    len_predict = 120
    ret = torch.zeros(1,len_predict,self.vocab_size) 

    if USE_GPU:
      ret = ret.to("cuda")
    inp_one_len = id_array[:,0]
    inp = inp_one_len.unsqueeze(0).transpose(0,1)
    emb = self.emb(inp)
    out, (hidden, cell) = self.lstm(emb)
    out = self.lin(out.squeeze(1))
    ret[:,1] = out
    top1id = out.argmax(1)
    inp_one_len = top1id

    for l in range(2,len_predict):
      inp = inp_one_len.unsqueeze(0).transpose(0,1)
      emb = self.emb(inp)
      out, (hidden, cell) = self.lstm(emb,(hidden,cell))
      out = self.lin(out.squeeze(1))
      ret[:,l] = out
      top1id = out.argmax(1)
      inp_one_len = top1id
    print(ret.size())
    return ret.argmax(2)
