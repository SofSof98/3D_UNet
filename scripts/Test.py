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

def test(opt):
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
    #train = [0,1]
    #validation = [2,3]
    #test = [3]
    test_patients = [files[i] for i in test]
    testing = dict(test = test_patients)
    test_loader = rd.DataLoader(opt,list_IDs=testing['test'], directory=opt.data_dir,
                                 dtype = opt.image_type,norm=opt.normalize,augmentation=None,
                                 n_channels=opt.n_channels,train=True)

     # Create folders and files.
    utils.create_folder(results_path)
    utils.store_results(results_path, opt,train = False)
    
    # Write options into a file.
    with open(results_path + '/test_options.txt', 'w') as f:
        f.write(" ".join(sys.argv[1:]))

    print("# train images = {}, # val images = {}, # test images = {}".format(len(train), len(validation), len(test)), end="\n\n")
    print("train images = \t{}".format(train))
    print("val images = \t{}".format(validation))
    print("test images = \t{}".format(test))

    print('loading data .....')
    
    x_test, mask_test = test_loader.Loading()



    num_classes = 1
    input_shape = opt.dim 
    output_shape = opt.dim 
    #model = UNet(input_shape, input_shape, num_classes, opt)
    model = model.Unet_3D(input_shape, output_shape, opt, num_classes=num_classes)
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    overall_test_loss = []
    detailed_test_loss = [[],[],[],[],[],[],[],[],[]]
    
    model = Training_Model(opt,model, train=False)
    
    complete_computation_times = []
    forward_computation_times = []
    print("started testing...")
    with torch.no_grad():
        for idx in range(len(test)):

            
            data, lesion = x_test[idx], mask_test[idx]
            p_id = test_patients[idx]
        
            model.validate(data, lesion)  
            prediction = model.get_prediction().cpu().detach().numpy()
            detailed_test_loss[0].append(model.get_loss())
            detailed_test_loss = utils.compute_losses(prediction, lesion.squeeze().cpu().detach().numpy(), detailed_test_loss)
            detailed_test_loss[8].append(model.get_iou_score(lesion))

            model.print_test_stats(idx, p_id, detailed_test_loss[1][-1], detailed_test_loss[8][-1])

         

            with open(results_path + "/test_results_detailed.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([idx, detailed_test_loss[0][-1], detailed_test_loss[1][-1], detailed_test_loss[8][-1], detailed_test_loss[2][-1],\
                    detailed_test_loss[3][-1], detailed_test_loss[4][-1], detailed_test_loss[5][-1], detailed_test_loss[6][-1], detailed_test_loss[7][-1], p_id])
    
            for i in range(np.asarray(detailed_test_loss).shape[0]):
                overall_test_loss.append(np.mean(detailed_test_loss[i]))

            # Writing overall test loss in file.
        with open(results_path + "/test_results.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([np.mean(detailed_test_loss[0]), np.mean(detailed_test_loss[1]), np.median(detailed_test_loss[1]), np.mean(detailed_test_loss[8]), np.median(detailed_test_loss[8]), np.mean(detailed_test_loss[2]), np.median(detailed_test_loss[2]),\
            np.mean(detailed_test_loss[3]), np.median(detailed_test_loss[3]), np.mean(detailed_test_loss[4]), np.mean(detailed_test_loss[5]), np.mean(detailed_test_loss[6]), np.median(detailed_test_loss[6]),\
            np.mean(detailed_test_loss[7]), np.median(detailed_test_loss[7]), np.std(detailed_test_loss[1]), np.std(detailed_test_loss[2]), np.std(detailed_test_loss[3]), np.std(detailed_test_loss[4]),\
            np.std(detailed_test_loss[5]), np.std(detailed_test_loss[6]), np.std(detailed_test_loss[7])])

        print("Completed testing in with\n overall cross entropy loss: {}, dice coefficient: {}".format(overall_test_loss[0], overall_test_loss[1]))


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
    test(opt)
