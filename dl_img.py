from fastprogress import master_bar, progress_bar
from fastai.vision import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks import *
import pandas as pd
from torch import optim
import re
import torch
from fastai import *
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import pretrainedmodels
from collections import OrderedDict
import math
import argparse
import torchvision
import pprint
from utils import *
#from models import *

#path = Path('/home/wb/下载/images/idcard/idcard1')
#dest = pdir.data/'data/idcard1'

#path = Path('/home/wb/下载/images/idcard/idcard-n')
#dest = pdir.data/'data/idcard2'

#path = Path('/home/wb/下载/images/yinhangka')
#dest = pdir.data/'data/bankcard'

path = Path('/home/wb/下载/images/creditcard')
dest = pdir.data/'data/creditcard'

download_images(path, dest, max_pics=200)