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
    name = f'{config.task.name}-{config.model.backbone}'

    img_size = config.train.img_size
    batch_size = config.train.batch_size
    model_file = config.train.model_file

    data = load_dump('data_bunch.dump')

    learn = unet_learner(data, models.resnet34, metrics=acc_camvid, wd=1e-2, model_dir=pdir.models)

    #learn.load('Segmentation-resnet34_9')
    #learn.save('Segmentation-noopt', with_opt=False)
    learn.load('Segmentation-noopt')

    path = pdir.data/'data/objects'
    file_name = path/'kettle2.jpg'
    file_name = path/'kettle1.jpg'

    #img = open_image(file_name)
    img = PIL.Image.open(file_name).convert('RGB')
    img.show()

    img = img.resize((224, 224))
    img_t = pil2tensor(img, np.float32)
    img_t = (img_t/255).to(device)
    img_t = normalize(img_t.double(), torch.tensor(imagenet_means).to(device), torch.tensor(imagenet_std).to(device))

    with torch.no_grad():
        result = learn.model(img_t.unsqueeze(0).float())
        mask = (torch.sigmoid(result[0][1])>0.5)

        mask *= 255
        im2 = PIL.Image.fromarray(mask.detach().cpu().numpy())
        #im2.show()
        im2.save('mask.png')

    #clean image, remove small points
    image = cv2.imread('mask.png')
    mask = cv2.imread('mask.png', cv2.IMREAD_GRAYSCALE)
    mask = clean_img(mask)

    #find minimum bounding box
    length, width = cal_min_area_rect(mask, image)
    print(length, width)




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
        args.config_file = 'yaml/seg-224.yml'

    config = load_config(args.config_file)
    pprint.PrettyPrinter(indent=2).pprint(config)
    #utils.prepare_train_directories(config)
    run(config)
    print('success!')


if __name__ == '__main__':
    main()



