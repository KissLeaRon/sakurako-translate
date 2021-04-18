import pandas as pd
import numpy as np
from hyper_params import * 
import torch
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data import DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence as pad

def word_dict(df):
  whole_copus = " ".join(df.old) + " " + " ".join(df.new)
  vocab = set(whole_copus.split(" "))
 
  count = EOS + 1
  word2id = dict()
  id2word = dict()
  for w in vocab:
    word2id[w] = count
    id2word[count] = w
    count += 1
  word2id["[BOS]"] = BOS
  word2id["[EOS]"] = EOS
  word2id["[UNK]"] = UNKNOWN
  id2word[BOS] = "[BOS]"
  id2word[EOS] = "[EOS]"
  id2word[UNKNOWN] = "[UNK]"
  id2word[0]   = "_"

  return word2id, id2word

def withUnknown(seq,word2id):
  ret = []
  for s in seq:
    id_array = [BOS]
    for token in s.split(" "):
      try:
        i = word2id[token]
      except KeyError:
        i = UNKNOWN
      id_array.append(i)
    id_array += [0]*10
    id_array.append(EOS)
    ret.append(id_array)
  return ret

def id2wordWithUnknown(twoD_array,id2word):
  ret = []
  for id_array in twoD_array.tolist():
    tokens = []
    for idx in id_array:
      tokens.append(id2word[idx])
    ret.append("".join(tokens))
  return ret


def df2onehot(df_train,df_test,word2id):
  old = withUnknown(df_train.old, word2id)
  new = withUnknown(df_train.new, word2id)
  old_tes = withUnknown(df_test.old, word2id)
  new_tes = withUnknown(df_test.new, word2id)
  return (old, new), (old_tes, new_tes) 

def data2dataset(w2i = None, i2w = None):
  df = pd.read_csv(PATH_DATA)
  if DEBUG_MODE:
    df = df[:32]
  df_dict, df_non_dict = train_test_split(df,test_size=0.2) 
  word2id, id2word = word_dict(df_dict)
  df = pd.concat([df_dict,df_non_dict])

  df_train, df_test = train_test_split(df,test_size=0.1) 
  train_onehot, test_onehot = df2onehot(df_train,df_test,word2id)
  dataset_train = MyDataset(*train_onehot,dict_len=len(word2id))
  dataset_test = MyDataset(*test_onehot,dict_len=len(word2id))
  return (dataset_train, dataset_test), (word2id,id2word)

def table2dl(w2i = None, i2w = None):
  (train,test),(w2i,i2w) = data2dataset(w2i=w2i,i2w=i2w)
  loader_train = DataLoader(train,collate_fn=train.collate,batch_size=BATCH_SIZE,shuffle=True)
  loader_test = DataLoader(test,collate_fn=test.collate,batch_size=BATCH_SIZE,shuffle=True)
  
  return loader_train, loader_test,(w2i,i2w)

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, old, new, dict_len):
    self.new = new
    self.len_new = [len(arr) for arr in new]
    if FINE_TUNING:
      self.old = old
      self.len_old = [len(arr) for arr in old]
    else:
      self.old = new
      self.len_old = [len(arr) for arr in new]
    self.vocab_size = dict_len

  def __len__(self):
    return len(self.old)

  def __getitem__(self, index):
    return {
        "old" : self.old[index],
        "new" : self.new[index],
        "len_old" : self.len_old[index],
        "len_new" : self.len_new[index]
        }

  def collate(self,items):
    old = [torch.tensor(i["old"]) for i in items]
    new = [torch.tensor(i["new"]) for i in items]
    len_old = [i["len_old"] for i in items]
    len_new = [i["len_new"] for i in items]
    return {
        "old" : pad(old,batch_first=True),
        "new" : pad(new,batch_first=True),
        "len_old" : torch.tensor(len_old),
        "len_new" : torch.tensor(len_new)
        }
