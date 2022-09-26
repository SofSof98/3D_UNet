import numpy as np
import nrrd
import medpy
import shutil
import os
import  csv
import utils.metrics as metrics
import torch
import sys
import json
import matplotlib.pyplot as plt
from medpy.metric.binary import hd, dc, asd, assd, precision,\
    sensitivity, specificity

# Some of the functions used here are borrowed from https://gitlab.com/dejankostyszyn/prostate-gtv-segmentation/-/tree/master
## add your meand and sd in the function
def cut_pet_old(opt,pet_array, prostate_array, pitch=5, new_size=64,
            normalization='local', n_channel = 1, train=False, concat=False):

 	
    if  n_channel > 1 and opt.rs == 2020:
        std = 6.211471196036908
        # Arithmetic mean to manually change.
        mean = 1.8105808395450398
    elif  n_channel > 1 and  opt.rs == 2021:
        std = 6.947726866705141
        # Arithmetic mean to manually change.
        mean = 1.8001914074865437
    elif  n_channel > 1 and opt.rs == 2022:
        std = 7.296833011765431
        mean = 1.8988180241342318
    elif  n_channel == 1 and opt.rs == 2020:
        mean =  0.10223833057966272
        std= 0.9130426563428523

    prostate = np.where(prostate_array == 1)
    # 3D array (x,y,z) for prostate segmenatation
    x = prostate[0]
    y = prostate[1]
    z = prostate[2]

    min_x = np.min(x) - pitch
    max_x = np.max(x) + pitch
    min_y = np.min(y) - pitch
    max_y = np.max(y) + pitch
    min_z = np.min(z) - pitch
    max_z = np.max(z) + pitch

    if (max_x - min_x) % 2 != 0:
        max_x += 1
    if (max_y - min_y) % 2 != 0:
        max_y += 1
    if (max_z - min_z) % 2 != 0:
        max_z += 1

    offset_x = int((new_size - (max_x - min_x)) / 2)
    offset_y = int((new_size - (max_y - min_y)) / 2)
    offset_z = int((new_size - (max_z - min_z)) / 2)

    max_x = max_x + offset_x
    min_x = min_x - offset_x
    max_y = max_y + offset_y
    min_y = min_y - offset_y
    max_z = max_z + offset_z
    min_z = min_z - offset_z

    pet_cut = pet_array[min_x:max_x, min_y:max_y, min_z:max_z]
    pet_cut = np.float32(pet_cut)
    if normalization == 'local':
        pet_cut = (pet_cut - pet_cut.min()) / (pet_cut.max() - pet_cut.min())
    elif normalization == 'global':
        pet_cut = (pet_cut - mean) / sd
	
    if concat:
        prostate_cut = prostate_array[min_x:max_x, min_y:max_y, min_z:max_z]
        return  np.float32(pet_cut),np.float32(prostate_cut)
    # pet = tf.convert_to_tensor(pet_cut, dtype = 'float32')
    # print(tf.shape(pet))
    else:
        return  np.float32(pet_cut)

def cut_pet(opt,pet_array, prostate_array, pitch=5, new_size=64,
            normalization='local', n_channel = 1, train=False, concat=False, prediction=False):

    if  n_channel > 1 and opt.rs == 2020:
        std = 6.211471196036908
        # Arithmetic mean to manually change.
        mean = 1.8105808395450398
    elif  n_channel > 1 and  opt.rs == 2021:
        std = 6.947726866705141
        # Arithmetic mean to manually change.
        mean = 1.8001914074865437
    elif  n_channel > 1 and opt.rs == 2022:
        std = 7.296833011765431
        mean = 1.8988180241342318   
    elif  n_channel == 1 and opt.rs == 2020:
        mean =  0.10223833057966272
        std= 0.9130426563428523
    prostate = np.where(prostate_array == 1)

    # 3D array (x,y,z) for prostate segmenatation
    x = prostate[0]
    y = prostate[1]
    z = prostate[2]

    min_x = np.min(x) - pitch
    max_x = np.max(x) + pitch
    min_y = np.min(y) - pitch
    max_y = np.max(y) + pitch
    min_z = np.min(z) - pitch
    max_z = np.max(z) + pitch

    if (max_x - min_x) % 2 != 0:
        max_x += 1
    if (max_y - min_y) % 2 != 0:
        max_y += 1
    if (max_z - min_z) % 2 != 0:
        max_z += 1

    limit = 64
    while max_x - min_x < limit:
        max_x += 1
        if min_x == 0:
            max_x += 1
        else:
            min_x -= 1

    while max_y - min_y < limit:
        max_y += 1
        if min_y == 0:
            max_y += 1
        else:
            min_y -= 1

    while max_z - min_z < limit:
        max_z += 1
        if min_z == 0:
            max_z += 1
        else:
            min_z -= 1
    pet_cut = pet_array[min_x:max_x, min_y:max_y, min_z:max_z]
    pet_cut =  np.float32(pet_cut)
    old_pet_shape = (pet_array.shape, min_x, max_x, min_y, max_y, min_z, max_z)

    if normalization == 'local':
        pet_cut = (pet_cut - pet_cut.min()) / (pet_cut.max() - pet_cut.min())
    elif normalization == 'global':
        pet_cut = (pet_cut - mean) / std

    if prediction and concat:
        prostate_cut = prostate_array[min_x:max_x, min_y:max_y, min_z:max_z]
        return  np.float32(pet_cut),np.float32(prostate_cut), old_pet_shape

    # pet = tf.convert_to_tensor(pet_cut, dtype = 'float32')
    # print(tf.shape(pet))

    elif concat:
        prostate_cut = prostate_array[min_x:max_x, min_y:max_y, min_z:max_z]
        return  np.float32(pet_cut),np.float32(prostate_cut)
        
    elif prediction:
        return  np.float32(pet_cut),old_pet_shape
    else:
        return  np.float32(pet_cut)


def store_results(results_path, opt,train = True):

    if train:
        with open(results_path + "/training_results.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch", "iterations", opt.loss+" loss", "img idx", "patient id"])

        with open(results_path + "/training_results_summary.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch", "iterations", opt.loss+" loss", "dice score"])

        with open(results_path + "/validation_results.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch", "patient idx", opt.loss+" loss", "dice coefficient","IoU coefficient",\
                    "hausdorff distance", "average surface distance metric",\
                    "average symmetric surface distance", "precison", "sensitivity",\
                    "specificity", "patient id"])
                
        with open(results_path + "/validation_results_summary.csv", "w", newline="") as file:
                writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(["epoch", opt.loss+" loss", "dice coefficient", "IoU coefficient",\
                    "hausdorff distance", "average surface distance metric",\
                    "average symmetric surface distance", "precison", "sensitivity",\
                    "specificity"])
    else:

        with open(results_path + "/test_results.csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["mean  loss", "mean DSC", "median DSC", "mean IoU", "median IoU",\
                "mean HD", "median HD", "mean average surface distance", "median average surface distance",\
                "mean average symmetric surface distance", "mean precison", "mean sensitivity", "median sensitivity",\
                "mean specificity", "median specificity", "std DSC", "std HD", "std ASD", "std ASSD", "std precision", "std sensi", "std speci"])
        with open(results_path + "/test_results_detailed.csv", "w", newline="") as file:
            writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["img idx", "loss", "dice coefficient","IoU coefficient",\
                "hausdorff distance", "average surface distance",\
                "average symmetric surface distance", "precison", "sensitivity",\
                "specificity", "patient id"])


  
def save_rewards(path,bel,loss, bed, dice, ber, ratio):
    json.dump(
        {
            "best epoch loss": bel,
            "best loss": float(loss),
            "best epoch dice": bed,
            "best dice": float(dice),
            "best epoch ratio": ber,
            "best ratio": float(ratio)

        },
        open(os.path.join(path,"results.json"), "w")
    )

def read_and_cut_files(path, prostate_path, normalization = 'none'):
    file, _ = nrrd.read(path)
    prostate, _ = nrrd.read(prostate_path)
    new_file = cut_pet(file, prostate, normalization = normalization, train = True)
    return new_file


def divide_data_num(patients, train, validation, test, shuffle=True):
    # 66 patients for training
    # 10 patients for validation
    # 10 patients for testing
    train = train / len(patients)
    val = train + (validation / len(patients))
    test = val + (test / len(patients))
    idx = np.arange(len(patients))

    # Planting a seed for reproducibility.
    np.random.seed(2022)

    if shuffle == True:
        np.random.shuffle(idx)

    return idx[:int(len(idx) * train)], idx[int(len(idx) * train):int(len(idx) * val)], idx[int(len(idx) * val):]


def divide_data(max_idx, opt):
    """
    This method shuffles the data and then divides it into
    70% for training
    10% for validation
    20% for testing
    """
    idx = np.arange(max_idx)

    # Planting a seed for reproducibility.
    np.random.seed(opt.rs)
    torch.manual_seed(opt.rs)
    # if torch.cuda.is_available():
      # torch.backends.cudnn.deterministic = True
      #torch.backends.cudnn.benchmark = False

    if opt.shuffle == True:
        np.random.shuffle(idx)
        
    return idx[:int(len(idx)*0.7)], idx[int(len(idx)*0.7):int(len(idx)*0.8)], idx[int(len(idx)*0.8):]


def arg_boolean(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("{} is not a valid boolean value. Please use one out of {}".format(value, t + f))



def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder(s) {}".format(path))
    else:
        print("Folder(s) {} already exist(s).".format(path))

def overwrite_request(path):
    
    if os.path.exists(path):
        valid = False
        answer = input("{} already exists. Are you sure you want to delete this folder? [yes/no]\n".format(path))
        
        if arg_boolean(answer):
                shutil.rmtree(path)
        
        elif not arg_boolean(answer):
            while not valid:
                answer = input("{} already exists. Are you sure you want to overwrite everything in this folder? [yes/no]\n".format(path))
                
                if arg_boolean(answer):
                    valid = True
                elif not arg_boolean(answer):
                    sys.exit(1)
            

def save_checkpoint(results_path, epoch, filename, best_epoch, best_dice, best_epoch_loss,
                        best_val_loss, model, optimizer):
        path = os.path.join(results_path,"ckpt")
        os.makedirs(path, exist_ok=True)
        filename = filename + ".ckpt"
        try:
            torch.save({'epoch': epoch,
                        'best_epoch': best_epoch,
                        'best_epoch_dc_score': best_dice,
                        'best_epoch_loss': best_epoch_loss,
                        'best_val_loss': best_val_loss,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                        },os.path.join(path, filename))

        except Exception as e:
            print("An error occurred while saving the checkpoint:")
            print(e)


def get_scheduler(optimizer, opt):

    if opt.scheduling_lr.lower() == 'reducelronplateau':
        print('scheduler=ReduceLROnPlateau')
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, threshold=1e-5, patience=50)
    elif opt.scheduling_lr.lower()  == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=0.5)
    elif opt.scheduling_lr.lower() == 'lrscheduler':
        print('scheduler=LRScheduler')
        steps = opt.steps

        if len(steps) == 2:
            
            def lambda_rule(epoch):
         
                if epoch < steps[0]:
                    lr_l = 1
                elif steps[0] <= epoch < steps[1]:
                    lr_l = 0.1
              
                elif steps[1] <= epoch:
                    lr_l = 0.01
                return lr_l
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        elif len(steps) == 3:

            def lambda_rule(epoch):
         
                if epoch < steps[0]:
                    lr_l = 1
                elif steps[0] <= epoch < steps[1]:
                    lr_l = 0.1
                elif steps[1] <= epoch < steps[2]:
                    lr_l = 0.01
                elif steps[2] <= epoch:
                    lr_l = 0.001
                return lr_l
        
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        else:
            sys.exit("{} is not valid.Either length 2 or 3".format(steps))
    else:

        return NotImplementedError('learning rate policy [%s] is not implemented', opt.scheduling_lr)

    return scheduler

def update_learning_rate(scheduler, metric=None, epoch=None):
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metrics=metric)
        else:
            scheduler.step()
        #lr = optimizer[0].param_groups[0]['lr']
        #print('current learning rate = %.7f' % lr)

        
def set_loss(opt, device):
    if opt.loss.lower() not in ["binary_cross_entropy", "dice", "dice_bce", "iou","tversky","focal_tv"]:
        sys.exit('Value for loss_fn must be either of {"binary_cross_entropy", "dice", "dice_bce", "iou","tversky"}')
    if opt.loss == "binary_cross_entropy":
        loss_fn = torch.nn.BCELoss()
    elif opt.loss == "dice":
        loss_fn = metrics.dice_loss
    elif opt.loss.lower() == "dice_bce":
        loss_fn = metrics.DiceBCELoss
    elif opt.loss.lower() == "iou":
        loss_fn = metrics.IoU_loss
    elif opt.loss.lower() == "tversky":
        loss_fn = metrics.Tversky_loss
    elif  opt.loss.lower() == "focal_tv":
        loss_fn = metrics.focal_tv_loss
    return loss_fn

def set_optimizer(opt, model_params):
    if opt.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model_params, lr=opt.lr_adam, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == "adamax":
        optimizer = torch.optim.Adamax(model_params, lr=lr_adamax, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    elif opt.optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adamax(model_params, lr=lr_adagrax, betas=(opt.beta1, opt.beta2), eps=opt.eps_adagrad, weight_decay=opt.weight_decay)

    elif opt.optimizer.lower() == "sgd":
        optimizer = torch.optim.Adamax(model_params, lr=lr_SDG, momentum=opt.momentum_SDG, weight_decay=opt.weight_decay, nesterov=opt.nesterov)
   
    else:
        sys.exit("{} is not a valid optimizer. Choose one of {adam, adamax}".format(opt.optimizer))
    return optimizer

# Compute validation losses.
def compute_losses(input, target, losses):

    if 0 == np.count_nonzero(input) and 0 == np.count_nonzero(target):
        l2 = 1.0
        l3 = 0.0
        l4 = 0.0
        l5 = 0.0
    elif 0 != np.count_nonzero(input) and 0 == np.count_nonzero(target):
        l2 = 0.0
        l3 = 50.0
        l4 = 25.0
        l5 = 25.0
    elif 0 == np.count_nonzero(input) and 0 != np.count_nonzero(target):
        l2 = 0.0
        l3 = 50.0
        l4 = 25.0
        l5 = 25.0
    else:
        l2 = dc(input, target) # Dice Coefficient.
        l3 = hd(result=input, reference=target) # Hausdorff Distance.
        l4 = asd(result=input, reference=target) # Average surface distance metric.
        l5 = assd(result=input, reference=target) # Average symmetric surface distance.
    l6 = precision(result=input, reference=target) # Precison.
    l7 = sensitivity(result=input, reference=target) # Sensitivity/ recall/ true positive rate.
    l8 = specificity(result=input, reference=target) # Specificity/ true negative rate.
    
    losses[1].append(l2)
    losses[2].append(l3)
    losses[3].append(l4)
    losses[4].append(l5)
    losses[5].append(l6)
    losses[6].append(l7)
    losses[7].append(l8)
    return losses

# Computes and returns Binary Hausdorff Distance of two tensors.
def hausdorff_distance(input, target):
    return hd(result=input, reference=target, voxelspacing=None, connectivity=1)

# Computes and returns dice coefficient of tensors.
def dice_coefficient(input, target):
    return dc(result=input, reference=target)

# Average surface distance metric.
def average_surface_distance_metric(input, target):
    return asd(result=input, reference=target)

# Average symmetric surface distance.
def average_symmetric_surface_distance(input, target):
    return assd(result=input, reference=target)

# Precison.
def m_precision(input, target):
    return precision(result=input, reference=target)

# True positive rate.
def m_sensitivity(input, target):
    return sensitivity(result=input, reference=target)

# True negative rate.
def m_specificity(input, target):
    return specificity(result=input, reference=target)

# Compute validation losses.
def compute_losses(input, target, losses):

    if 0 == np.count_nonzero(input) and 0 == np.count_nonzero(target):
        l2 = 1.0
        l3 = 0.0
        l4 = 0.0
        l5 = 0.0
    elif 0 != np.count_nonzero(input) and 0 == np.count_nonzero(target):
        l2 = 0.0
        l3 = 50.0
        l4 = 25.0
        l5 = 25.0
    elif 0 == np.count_nonzero(input) and 0 != np.count_nonzero(target):
        l2 = 0.0
        l3 = 50.0
        l4 = 25.0
        l5 = 25.0
    else:
        l2 = dc(input, target) # Dice Coefficient.
        l3 = hd(result=input, reference=target) # Hausdorff Distance.
        l4 = asd(result=input, reference=target) # Average surface distance metric.
        l5 = assd(result=input, reference=target) # Average symmetric surface distance.
    l6 = precision(result=input, reference=target) # Precison.
    l7 = sensitivity(result=input, reference=target) # Sensitivity/ recall/ true positive rate.
    l8 = specificity(result=input, reference=target) # Specificity/ true negative rate.

    losses[1].append(l2)
    losses[2].append(l3)
    losses[3].append(l4)
    losses[4].append(l5)
    losses[5].append(l6)
    losses[6].append(l7)
    losses[7].append(l8)
    #losses[8].append(0)
    return losses

def plot_losses(opt, path, title, xlabel, ylabel, plot_name, *args, axis="auto"):
    """
    Creates nice plots and saves them as PNG files onto permanent memory.
    """
    fig = plt.figure()
    plt.title(title)
    for element in args:
        plt.plot(element[0], label=element[1], alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.axis(axis)
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(path, plot_name))
    plt.close(fig)


def import_model_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    print(mod)
    return mod

