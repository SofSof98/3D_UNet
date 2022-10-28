import ReadData as rd
import numpy as np
import pandas as pd
from matplotlib import colors
import torch, os, nrrd, time
import matplotlib.pyplot as plt
import  csv, sys
import utils.Utils as utils
import utils.metrics as metrics
from models import *
from Training_Model import Training_Model
from os.path import join
import json

def fit_prediction_to_pet(pet_shape,  prediction, opt):
    t = np.full(pet_shape[0], 0, dtype=float)
    min_x, max_x, min_y, max_y, min_z,max_z = pet_shape[1:]
    t[min_x:max_x, min_y:max_y, min_z:max_z] = prediction

    return t

def prediction(opt):
    print("Loading the data ...")


    results_path = os.path.join(opt.results_dir)
    _, model_name = opt.model.split('.')
    name =  model_name + "_"  + str(opt.scheduling_lr) + "_"+ opt.optimizer + '_' + opt.loss  + '_' + str(opt.dr)
    ckpt_path = results_path + name + '/ckpt/' + 'best_' + opt.best_model + '_model.ckpt'
    #name = model_name + "_"  + str(opt.scheduling_lr) + "_"+ opt.optimizer + '_' + opt.loss + '_' + str(opt.dr)
    results_path = join(results_path, name, 'test_' + opt.best_model)
    #utils.overwrite_request(results_path)
    
    model = utils.import_model_class(opt.model)
   


    txt_root = opt.txt_dir

    with open(txt_root) as file:
        files = [x.strip() for x in file.readlines()]

   
    train, validation, test = utils.divide_data(len(files), opt)
    test_patients = [files[i] for i in test]
 
    
     # Create folders and files.
    utils.create_folder(results_path)
    
    
 
   


    num_classes = 1
    input_shape = opt.dim 
    output_shape = opt.dim 
    #model = UNet(input_shape, input_shape, num_classes, opt)
    model = model.Unet_3D(input_shape, output_shape, opt, num_classes=num_classes)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    
    model = Training_Model(opt,model, train=False)
    forward_computation_times = [] 
    
    print("started testing...")
    with torch.no_grad():
        for idx in test_patients:
             
            print(idx)

            loader = rd.PatientLoader(opt,idx, directory=opt.data_dir,
                                 dtype = opt.image_type,norm=opt.normalize,augmentation=None,
                                 n_channels=opt.n_channels,train=True)
		
            data, pet_shape, pet_header  = loader.Loading()
            print(pet_shape)
            fw_pass_start = time.time()
            model.forward(data)
            fw_pass_end = time.time()
            prediction = model.get_prediction().cpu().detach().numpy()
            prediction = fit_prediction_to_pet(pet_shape, prediction, opt)
            prediction_name = os.path.join(results_path,idx + '_pred')            
            nrrd.write(prediction_name, prediction, pet_header)
            print("Saved as {}".format(prediction_name))
            forward_computation_times.append(fw_pass_end - fw_pass_start)
        print(np.mean(forward_computation_times))
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
    print('Prediction...')
    prediction(opt)
