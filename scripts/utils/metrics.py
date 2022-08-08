import numpy as np
import matplotlib.pyplot as plt
import os, torch, medpy, sys, torch, random
from medpy.metric.binary import hd, dc, asd, assd, precision,\
    sensitivity, specificity
import torch.nn.functional as F


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

    return losses

# Dice Loss (only for 2 labels)
def dice_coeff(y_pred, y_true):
    smooth = 0.000001

    # Flatten
    # y_true_f = torch.reshape(y_true, (-1,))
    #y_pred_f = torch.reshape(y_pred, (-1,))
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    
    intersection = torch.sum(y_true_f * y_pred_f)
    score = 2. * intersection / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_pred, y_true):
    loss = 1 - dice_coeff(y_pred, y_true)
    return loss

# DiceBCELoss

def DiceBCELoss(y_pred, y_true):
    BCE = F.binary_cross_entropy(y_pred, y_true.squeeze(), reduction='mean')
    dice = dice_loss(y_pred, y_true)
    Dice_BCE_loss = BCE + dice
    return Dice_BCE_loss

# Tversky_Loss

def Tversky_coeff(y_pred, y_true):
    alpha = 0.3
    beta = 0.7
    smooth = 0.000001
    # Flatten
    y_true_f = torch.reshape(y_true, (-1,))
    y_pred_f = torch.reshape(y_pred, (-1,))
    #y_true_f = y_true.view(-1)
    #y_pred_f = y_pred.view(-1)
    
       
    #True Positives, False Positives & False Negatives
    TP = (y_pred * y_true).sum()    
    FP = ((1-y_true) * y_pred).sum()
    FN = (y_true * (1-y_pred)).sum()
       
    Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  
        
    return Tversky
    
def Tversky_loss(y_pred, y_true):
    loss = 1 - Tversky_coeff(y_pred, y_true)
    return loss

def focal_tv_loss(y_pred, y_true):
    gamma = 4/3
    loss = (1 - Tversky_coeff(y_pred, y_true))**gamma
    return loss

# IoU Loss

def IoU_coeff(y_pred, y_true):
    smooth = 0.000001

    # Flatten
    y_true_f = torch.reshape(y_true, (-1,))
    y_pred_f = torch.reshape(y_pred, (-1,))
    #y_true_f = y_true.view(-1)
    #y_pred_f = y_pred.view(-1)
     
    #intersection is equivalent to True Positive count
    #union is the mutually inclusive area of all labels & predictions 
    intersection = (y_pred * y_true).sum()
    total = (y_pred + y_true).sum()
    union = total - intersection 
        
    IoU = (intersection + smooth)/(union + smooth)
                
    return  IoU

def IoU_loss(y_pred, y_true):
    loss = 1 - IoU_coeff(y_pred, y_true)
    return loss


# plot losses and dice scores
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



