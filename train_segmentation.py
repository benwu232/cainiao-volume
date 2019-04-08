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


def run(config):
    name = f'{config.task.name}-{config.model.backbone}-{config.loss.name}'

    batch_size = config.train.batch_size

    if not voc_lbl.exists():
        gen_new_lbl(voc_ori_lbl, voc_lbl)
    #exit()

    lbl_names = get_image_files(voc_lbl)

    #mask_list = []
    #for ln in lbl_names:
    #    mask = open_mask(ln)
    #    mask_list.extend(mask.data.unique().tolist())
    #mask_list = sorted(list(set(mask_list)))


    path_info = path_voc/'ImageSets/Segmentation'
    all_seg = pd.read_csv(path_info/'trainval.txt', header=None)
    all_seg.columns = ['file']
    all_seg = df_stem2file(all_seg, 'jpg')


    val_seg = pd.read_csv(path_info/'val.txt', header=None)
    val_seg.columns = ['file']

    all_seg_list = list(range(len(all_seg)))
    random.shuffle(all_seg_list)
    val_idxes = all_seg_list[:400]

    src = (SegmentationItemList.from_df(all_seg, path=voc_img)
           .split_by_idx(val_idxes)
           .label_from_func(get_y_fn, classes=[0, 1])
          )

    data = (src.transform(get_transforms(), size=224, tfm_y=True)
           .databunch(bs=4, num_workers=NUM_WORKERS)
           .normalize(imagenet_stats))

    learn = unet_learner(data, models.resnet34, metrics=acc_camvid, wd=1e-2, model_dir=pdir.models)

    lr_find(learn)
    learn.recorder.plot()

    lr = 2e-3

    learn.fit_one_cycle(10, slice(lr), pct_start=0.9)

    learn.save('stage-1')



def parse_args():
    description = 'Train humpback whale identification'
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('-c', '--config', dest='config_file',
                        help='configuration filename',
                        default=None, type=str)
    return parser.parse_args()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('Training segmentation tasks for irregular objects ... ')
    args = parse_args()
    if args.config_file is None:
        #raise Exception('no configuration file')
        args.config_file = 'yaml/seg.yml'

    config = load_config(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    #utils.prepare_train_directories(config)
    run(config)
    print('success!')


if __name__ == '__main__':
    main()


