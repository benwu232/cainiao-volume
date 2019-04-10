import numpy as np
import torch
import pandas as pd
import torch
from copy import deepcopy
import datetime as dt
import time
import tqdm
import fastai
import cv2
from fastai.vision import *
from fastai.basic_data import *
from fastai.metrics import accuracy
from fastai.basic_data import *
from fastai.callbacks.hooks import num_features_model, model_sizes
import torchvision
import tensorboardX as tx
from common import *

PATH = './'
TRAIN = '../input/train/'
TEST = '../input/test/'
LABELS = '../input/train.csv'
BOXES = '../input/bounding_boxes.csv'
MODELS = './models'
SZ = 224
BS = 32
NUM_WORKERS = 0

path_voc = pdir.data/'voc/VOC2012'
voc_img = path_voc/'JPEGImages'
voc_ori_lbl = path_voc/'SegmentationClass'
voc_lbl = pdir.data/'voc_new_lbl'

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

#borrowed from  https://github.com/meetshah1995/pytorch-semseg
def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray(
        [
            [0, 0, 0],
            [128, 0, 0],
            [0, 128, 0],
            [128, 128, 0],
            [0, 0, 128],
            [128, 0, 128],
            [0, 128, 128],
            [128, 128, 128],
            [64, 0, 0],
            [192, 0, 0],
            [64, 128, 0],
            [192, 128, 0],
            [64, 0, 128],
            [192, 0, 128],
            [64, 128, 128],
            [192, 128, 128],
            [0, 64, 0],
            [128, 64, 0],
            [0, 192, 0],
            [128, 192, 0],
            [0, 64, 128],
        ]
    )

def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = (ii != 0)
    label_mask = label_mask.astype(int)
    return label_mask

def gen_new_lbl(ori_lbl_path, new_lbl_path):
    # conver masks and save them
    new_lbl_path.mkdir()
    lbl_names = get_image_files(ori_lbl_path)
    for lname in lbl_names:
        mask = open_mask(lname, convert_mode='RGB')
        # mask.show(alpha=1)
        # mask.data.unique()

        mask = mask.data.numpy().astype(int).transpose(1, 2, 0)
        new_mask = encode_segmap(mask)
        new_image = PIL.Image.fromarray(new_mask.astype(np.uint8))
        new_image.save(new_lbl_path / f'{lname.name}')

def stem2file(stem, suffix='jpg'):
    return f'{stem}.{suffix}'

def df_stem2file(df, suffix='jpg'):
    for k in range(len(df)):
        df.at[k, 'file'] = stem2file(df.loc[k, 'file'], suffix)
    return df

#get_y_fn = lambda x: voc_lbl/f'{Path(x).stem}.png'

def get_y_fn(fname, lbl_path=voc_lbl):
    return lbl_path/f'{Path(fname).stem}.png'

def acc_camvid(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def cv_imshow(img):
    cv2.imshow('Image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def clean_img(img):
    #mask_file.shape
    #mask = cv2.imread(str(img_file), cv2.COLOR_BGR2GRAY)
    #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # perform a series of erosions and dilations
    closed = cv2.erode(closed, None, iterations=4)
    closed = cv2.dilate(closed, None, iterations=4)

    return closed


def cal_min_area_rect(bin_img, ori_img=None):
    # finding contours
    cnts, _ = cv2.findContours(bin_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]

    # compute the rotated bounding box of the largest contour
    rect = cv2.minAreaRect(c)

    if ori_img is not None:
        #draw the minimum bounding box
        box = np.int0(cv2.boxPoints(rect))
        cv2.drawContours(ori_img, [box], -1, (0, 255, 0), 3)
        cv_imshow(ori_img)

    length, width = rect[1]
    if length < width:
        length, width = width, length
    return length, width




