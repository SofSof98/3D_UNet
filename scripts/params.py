import rd as rd
import numpy as np
import pandas as pd
from matplotlib import colors
import torch, os, nrrd, time
import matplotlib.pyplot as plt
import  csv, sys
import utils.Utils as utils
import utils.metrics as metrics
from models import *
from torchsummary import summary
from Training_Model import Training_Model
from os.path import join
import json

def train(opt):
    print("Loading the data ...")
    start_time = time.time()

    txt_root = opt.txt_dir

    with open(txt_root) as file:
        files = [x.strip() for x in file.readlines()]
        #print(files)


    # Divide data into training, validation and test set.
    train, validation, test = utils.divide_data(len(files), opt)
    #train = [0,1]
    #validation = [2,3]
    #test = [3]
    #train = [68,  7, 75]
    train_patients = [files[i] for i in train]
    validation_patients = [files[i] for i in validation]
    test_patients = [files[i] for i in test]

 
    training = dict(train = train_patients, validation = validation_patients,
                    test = test_patients)

    training_loader = rd.DataLoader(list_IDs=training['test'], directory=opt.data_dir,
                                 dtype = opt.image_type,norm=None,augmentation=False,n_channels=2, train=True)

    x_train, x_mask = training_loader.Loading()
    return x_train, x_mask
if __name__ == "__main__":

    import argparse
    import yaml
    import Parser as par

    parser = par.get_parser()
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.safe_load(f)
            print(default_arg)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    opt = parser.parse_args()
    print('Testing ...')
    x,y =train(opt)
    for i in range(len(y)):

        print('sum:',x[i].sum())

