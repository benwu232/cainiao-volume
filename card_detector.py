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
#from dataset import *


card_path = pdir.data

sub_dirs = ['bankcard', 'creditcard', 'idcard1', 'idcard2']
card_paths = [card_path/f'data/{name}' for name in sub_dirs]

image_file = str(card_paths[0]/'00000025.jpg')

image = cv2.imread(image_file)
cv_imshow(image)
thresh = preprocess_image(image)
cv_imshow(thresh)
pass