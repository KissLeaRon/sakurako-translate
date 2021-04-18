import torch
from torch import nn
import numpy as np

import read_csv
import model_making
import utils
from hyper_params import *
from tqdm import tqdm

utils.log_dirctory()
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir = PATH_LOG+"/log")

def init_weights(model):
  for name, param in model.named_parameters():
    nn.init.uniform_(param.data, -0.1,0.1)

################################################
###
#model and dataset loading

if FINE_TUNING:
  model, word2id, id2word = utils.load_models()
  train_dataloader, test_dataloader, (word2id, id2word)= read_csv.table2dl(w2i=word2id, i2w=id2word)
else:
  train_dataloader, test_dataloader, (word2id, id2word)= read_csv.table2dl()
  model = model_making.load_model(len(word2id))
  model.apply(init_weights)

optimizer, scheduler = model_making.optimizer_scheduler(model,len(train_dataloader))
criterion = nn.CrossEntropyLoss(ignore_index = 0)

example, lengths = utils.str2id(word2id)
###
#train
history_kappa_valid = []
history_kappa_test  = []
if USE_GPU:
  model = model.to("cuda")

for epoch in range(EPOCH):
  logger.info("EPOCH {}".format(epoch+1))
  model.train()
  total_train_loss = 0
  num_train_sample = 0
  for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
    num_train_sample += len(batch["len_old"])
    model.zero_grad()

    mask = model.generate_square_subsequent_mask(batch["new"].size(1))
    if USE_GPU:
      mask = mask.to("cuda")
    pred = model(batch["new"],mask,batch["old"].size()[1])  
    pred = pred.reshape(-1,pred.shape[-1])
    label = batch["old"].reshape(-1)
    if USE_GPU:
      label = label.to("cuda")
    loss = criterion(pred,label)
    total_train_loss += loss.item()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    scheduler.step()
    #if step % 10 == 0:
    #  print(step+1,"/",len(train_dataloader))
  model.eval()
  with torch.no_grad():
    mask = model.generate_square_subsequent_mask(example.size(1))
    ret = model(example,mask,len_label=None)
    print(read_csv.id2wordWithUnknown(ret,id2word))
    avg_train_loss = total_train_loss / num_train_sample
    logger.info("avg loss train: {}".format(avg_train_loss))
torch.save(model.state_dict(),"done/model_dict.pth")
import pickle
with open("done/word2id.pickle","wb") as f:
  pickle.dump(word2id,f)
with open("done/id2word.pickle","wb") as f:
  pickle.dump(id2word,f)
