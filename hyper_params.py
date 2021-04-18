###
#logger setting
from logging import getLogger, StreamHandler, Formatter, DEBUG
logger = getLogger("Logger")
handler = StreamHandler()
handler.setLevel(DEBUG)
fmt = Formatter("%(asctime)s %(levelname)s %(message)s")
handler.setFormatter(fmt)
logger.setLevel(DEBUG)
logger.addHandler(handler)
logger.propagate = False

###
#hyper parameters
from torch.cuda import is_available
import argparse
from datetime import date

parser = argparse.ArgumentParser()
parser.add_argument("--debug",help="debug mode",action="store_true")
parser.add_argument("-e","--epoch",help="Epoch",type=int,default=50)
parser.add_argument("-b","--batch",help="Batch size",type=int,default=8)
parser.add_argument("--learning_rate",help="Learning rate",type=float,default=1e-4)
parser.add_argument("--warming_up",help="epochs for warming up",type=int, default=5)
parser.add_argument("-d","--dropout",help="Dropout prob",type=float,default=0.1)
parser.add_argument("--fine_tuning",help="fine tuning",action="store_true")
args = parser.parse_args()

##
#hyper params
DEBUG_MODE = args.debug
FINE_TUNING = args.fine_tuning
if DEBUG_MODE:
  args.epoch = 2
EPOCH = args.epoch
BATCH_SIZE = args.batch
USE_GPU = is_available()
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
PATH_DATA = "data/wakati_pair.csv"
DROPOUT = args.dropout
LEARNING_RATE = args.learning_rate
WARMING_UP = args.warming_up


PATH_LOG = "log/"+date.today().isoformat()

###
#special tokens
UNKNOWN = 1
BOS = 2
EOS = 3

assert BATCH_SIZE > 0, "BATCH_SIZE must be > 0"
assert EPOCH > 0, "EPOCH must be > 0"

for k,i in args.__dict__.items():
  logger.info("{}\t:{}".format(k,i))
