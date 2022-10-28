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
from torchsummary import summary
from Training_Model import Training_Model
from os.path import join
import json

def train(opt):
    print("Loading the data ...")
    start_time = time.time()
    # Check if results folder exists and if so, ask user if really want to continue.
    results_path = os.path.join(opt.results_dir)
    _, model_name = opt.model.split('.')
    name = model_name + "_"  + str(opt.scheduling_lr) + "_"+ opt.optimizer + '_' + opt.loss + '_' + str(opt.dr)
    results_path = os.path.join(results_path, name)
    
    #utils.overwrite_request(results_path)

    model = utils.import_model_class(opt.model)
   
    txt_root = opt.txt_dir

    with open(txt_root) as file:
        files = [x.strip() for x in file.readlines()]
        #print(files)
    
    
    # Divide data into training, validation and test set.
    train, validation, test = utils.divide_data(len(files), opt)
    #train = [0,1]
    #validation = [2,3]
    #test = [3]
    train_patients = [files[i] for i in train]
    validation_patients = [files[i] for i in validation]
    test_patients = [files[i] for i in test]

    
    training = dict(train = train_patients, validation = validation_patients,
                    test = test_patients)

    training_loader = rd.DataLoader(opt,list_IDs=training['train'], directory=opt.data_dir,
                                 dtype = opt.image_type,norm=opt.normalize,augmentation=opt.augmentation,n_channels=opt.n_channels,
                                 train=True)

    print('loading data .....')
    x_train, mask_train =  training_loader.Loading()

    validation_loader = rd.DataLoader(opt,training['validation'],directory=opt.data_dir,
                                 dtype = opt.image_type,norm=opt.normalize,augmentation=False,n_channels=opt.n_channels,
                                 train=True)

    x_val, mask_val = validation_loader.Loading()

    nr_of_epochs = opt.epochs

    # Create folders and files.
    if opt.start_epoch == 0:
        utils.create_folder(results_path)
        utils.store_results(results_path, opt, train = True)
          
    # Write options into a file.
    with open(results_path + '/training_options.txt', 'w') as f:
        f.write(" ".join(sys.argv[1:]))
        
    num_classes = 1

    overall_train_loss = []
    overall_train_dice = []
    overall_val_loss = [[],[],[],[],[],[],[],[]]

    best_epoch = 0
    best_dice = 0.0
    best_val_loss = 1.0
    dice_best_val_loss = 0.0
    best_epoch_loss = 0
    best_ratio = 0.0
    best_epoch_ratio = 0
    

    print("# train images = {}, # val images = {}, # test images = {}".format(len(train), len(validation), len(test)), end="\n\n")
    print("train images = \t{}".format(train))
    print("val images = \t{}".format(validation))
    print("test images = \t{}".format(test))

    input_shape = opt.dim 
    output_shape = opt.dim 
    #model = model.Unet_3D(input_shape, output_shape, num_classes, opt)
    model = model.Unet_3D(input_shape, output_shape, opt, num_classes=num_classes)
    summary(model, input_size=(1,opt.n_channels,64,64,64))

    model = Training_Model(opt,model, train=True)


    for epoch in range(opt.start_epoch, nr_of_epochs):

        print("started training in epoch {} ...".format(epoch))
        train_losses = []
        train_dice = []

        for iteration, idx in enumerate(train,1):
            # Load data.
            data, lesion = x_train[iteration-1], mask_train[iteration-1]
            p_id = train_patients[iteration-1]

            if opt.batch_size == 1:
                model.optimize_parameters(data, lesion)
                
            elif opt.batch_size > 1:
                model.optimize_parameters_accumulate_grd(data, lesion, iteration, train)
            

            train_losses.append(model.get_loss())
            train_dice.append(model.get_score(lesion))

            model.print_stats(epoch, train_dice[-1], idx,iteration)

            with open(results_path + "/training_results.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, iteration, train_losses[-1], idx, p_id])

        utils.save_checkpoint(results_path, epoch, 'final_model', best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model.model, model.optimizer)

        # Intermediate saving of results.
        if (epoch+1) % opt.save_freq == 0 and epoch > 0:
            intermediate_model_name =  "_" + str(epoch + 1) + "_model"
            utils.save_checkpoint(results_path, epoch, intermediate_model_name, best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model.model, model.optimizer)
    
        overall_train_loss.append(np.mean(train_losses))
        overall_train_dice.append(np.mean(train_dice))

        with open(results_path + "/training_results_summary.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, np.mean(train_losses),np.mean(train_dice)])
                  
        print("finished training for epoch {}...".format(epoch))

        print("starting validation in epoch {} ...".format(epoch))
      
        val_losses = [[],[],[],[],[],[],[],[],[]]
        val_loss =[]
        with torch.no_grad():
            for idx in range(len(validation)):
                
                data, lesion = x_val[idx], mask_val[idx]
                p_id = validation_patients[idx]
        

                model.validate(data, lesion)
                val_losses[0].append(model.get_loss())
                prediction = model.get_prediction().cpu().detach().numpy()
                val_losses = utils.compute_losses(prediction, lesion.squeeze().cpu().detach().numpy(), val_losses)
                val_losses[8].append(model.get_iou_score(lesion))
                model.print_val_stats(epoch, val_losses[1][-1], val_losses[8][-1])
                with open(results_path + "/validation_results.csv", "a", newline="") as file:
                    writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    writer.writerow([epoch, idx, val_losses[0][-1], val_losses[1][-1], val_losses[8][-1], val_losses[2][-1],\
                        val_losses[3][-1], val_losses[4][-1], val_losses[5][-1], val_losses[6][-1], val_losses[7][-1], p_id])
                
        # update learning rate
        if opt.scheduling_lr is not None:
            model.update_lr
                
                
        with open(results_path + "/validation_results_summary.csv", "a", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow([epoch, np.mean(val_losses[0]), np.mean(val_losses[1]),np.mean(val_losses[8]),np.mean(val_losses[2]),\
                        np.mean(val_losses[3]), np.mean(val_losses[4]), np.mean(val_losses[5]), np.mean(val_losses[6]), np.mean(val_losses[7])])
        
        
        for i in range(8):
            overall_val_loss[i].append(np.mean(val_losses[i]))
        print("finished validation with mean loss = {}".format(np.mean(val_losses[0])))
        print("finished validation with mean dice score  = {}".format(np.mean(val_losses[1])))
        print("finished validation with mean IoU score = {}".format(np.mean(val_losses[8])))

        
        val_loss =np.mean(val_losses[0])
        val_dice = np.mean(val_losses[1])

        
        ## Saving the best models
        if  val_dice > best_dice:
            best_epoch = epoch
            best_dice = val_dice
            dice_model_name = "best_dice_model"
            utils.save_checkpoint(results_path, epoch, dice_model_name , best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model.model, model.optimizer)
            

        if  np.mean(val_losses[0]) < best_val_loss:
            best_val_loss = np.mean(val_losses[0])
            dice_best_val_loss = np.mean(val_losses[1])
            best_epoch_loss = epoch
            loss_model_name = "best_loss_model"
            utils.save_checkpoint(results_path, epoch, loss_model_name , best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model.model, model.optimizer)
        

        if  (val_dice/val_loss) > best_ratio:
            best_ratio  = val_dice/val_loss
            best_epoch_ratio = epoch
            ratio_model_name = "best_ratio_model"
            utils.save_checkpoint(results_path, epoch, ratio_model_name , best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model.model, model.optimizer)

       ## Plots
        utils.plot_losses(opt, results_path, "Training and Validation losses", "epochs",\
            "loss/dice", "losses.png", \
            [overall_train_loss, "train loss ("+opt.loss+")"],\
            [overall_val_loss[0], "val loss("+opt.loss+")"],\
            [overall_train_dice, "train (dice coeff) "],\
            [overall_val_loss[1], "val (dice coeff)"])

        utils.plot_losses(opt, results_path, "Hausdorff distance", "epochs",\
                "loss", "hausdorff_distance.png", \
                [overall_val_loss[2], "val (hd)"],\
                [overall_val_loss[3], "val (asd)"],\
                [overall_val_loss[4], "val (assd)"])
        utils.plot_losses(opt, results_path, "Different metrics", "epochs",\
            "loss", "other_metrics.png",\
            [overall_val_loss[5], "val (precision)"],\
            [overall_val_loss[6], "val (sensitivity)"],\
            [overall_val_loss[7], "val (specificity)"])
                
    utils.save_rewards(results_path,best_epoch_loss +1 , best_val_loss, dice_best_val_loss, best_epoch +1 , best_dice, best_epoch_ratio +1, best_ratio)
    end_time = time.time()
    print("Completed training and validation in {}s".format(end_time - start_time))
    print("Completed training with validation mean dice {} at epoch {}".format(best_dice, best_epoch + 1))
    print("Completed training with validation mean loss {} at epoch {}".format(best_val_loss, best_epoch_loss + 1))


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
    if opt.batch_size == 1:

        print('Training stochastic ...')
        train(opt)
    elif opt.batch_size > 1:
    
        print('Training in batches ...')
        train(opt)
    elif  opt.batch_size < 0:
            sys.exit("Batch size is not valid.Must be > 0")
 
