from hyper_params import *
import os
import pandas as pd
from read_csv import withUnknown
import torch

def log_dirctory():
  if os.path.isdir(PATH_LOG):
    logger.info(PATH_LOG+": exists!")
    #raise os.FileExistsError
  os.makedirs(PATH_LOG+"/summary",exist_ok=True)
  os.makedirs(PATH_LOG+"/log",exist_ok=True)
  logger.info("log dirctory created")
  return 0 


def log_summary(score):
  path = PATH_LOG+"/summary/summary_val"+str(VALIDATION)+".csv"
  try:
    df_summary = pd.read_csv(path,index_col=0)
  except FileNotFoundError:
    indices = args.__dict__keys()
    values = args.__dict__.values()
    df_summary = pd.DataFrame(values,index=indices,columns=["value"])
  df_score = pd.DataFrame(score,index=["kappa"+str(PROMPT)],columns=["value"])
  df_new = df_summary.append(df_score)
  df_new.to_csv(path)
  logger.info("output summary")
  return 0 

def log_avarage_in_validation():
  path = PATH_LOG+"/summary/summary_val"+str(VALIDATION)+".csv"
  df_summary = pd.read_csv(path,index_col=0)
  kappa = df_summary.query('index.str.contains("kappa")').astype(float)
  kappa = kappa.mean().item()
  df_score = pd.DataFrame(kappa,index=["avg"],columns=["value"])
  df_new = df_summary.append(df_score)
  df_new.to_csv(path)
  return 0

def str2id(word2id):
  example = "子供たち は みんな 、 仲 の いい 友達 と 思い思い に 連れ だって 、 祭り の にぎわい に 繰り出す 中 。 浴衣姿 の ダイヤ は キリリ と し た 風情 で 。 キュッと 口 を 引き 結ん で 1人 、 漁協 の 建物 正面 に 作ら れ た 白 テント の 下 の 貴賓 席 の パイプ椅子 に 座っ て た 。 まわり に は 大人 ばかり 。 内浦 でも 目立つ 大きな お 店 の ご 主人 や 、 町内会 の 会長 、 あ 、 あそこ に ぺこぺこ 頭 を 下げ てる 小学校 の 校長先生 も いる"
  #tagger = MeCab.Tagger("-Owakati")
  #parse = tagger.parse(example)[:-1]
  parse = example
  ret = withUnknown([parse],word2id)
  ret = torch.tensor(ret).long()
  length = torch.tensor(len(ret[0])).unsqueeze(0)
  return ret, length

def load_models():
  import pickle
  with open("done/word2id.pickle","rb") as f:
    word2id = pickle.load(f)
  with open("done/id2word.pickle","rb") as f:
    id2word = pickle.load(f)
  from self_attention import TransformerModel
  model = TransformerModel(len(word2id.keys())+1,64,8,64,2,DROPOUT)
  model.load_state_dict(torch.load("done/model_dict.pth"),strict=False)
  return model, word2id, id2word

if __name__ == "__main__":
  model, word2id, id2word = load_models()
  example =" 考え て き た 挨拶 を 順番 に 披露 し て いっ た 。 すぐ に ヨハネ の 番になって 、 アイドル らしく 振る舞おう と し た けど 。 台詞 を 口 に 出し ながら 同時 に 、 もっと 上手く 出来る はず って 、 仕草 一つ が 気 に なって しまって 。 一つ 言葉 を つっかえ て たら すぐ 、 場 が しん って 鎮まっ ちゃっ て 。 この 静寂 は 自分 が 作っ て る ん だ と 思っ た ら 頭 が 真っ白 に なっ て 、 前日 あれ だけ 練習 し て そら で 言え る よう に なった 台詞 も 喉 に 張り付いた 。"
  parse = example
  ret = withUnknown([parse],word2id)
  ret = torch.tensor(ret).long()
  length = torch.tensor(len(ret[0])).unsqueeze(0)
  mask = model.generate_square_subsequent_mask(ret.size(1))
  ret = model(ret,mask,len_label=None)
  import read_csv
  print(read_csv.id2wordWithUnknown(ret,id2word))

