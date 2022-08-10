import ReadData_new as rd
import numpy as np
import torch, nrrd, time, csv
import  util as utils
from options.train_options import TrainOptions
from os.path import join

def get_params():
    opt = TrainOptions().parse()
    opt.normalize = "None"
    vl = rd.ValLoader(opt)
    max_idx = vl.nr_of_patients()
    nr_of_elements = 0

    std = 0.0
    mean = 0.0
    idx = 0

    # Compute training, validation and test permutation.
    idx_train, idx_val, idx_test = utils.divide_data(max_idx, opt)
    pets = []
    masks = []
    # Compute mean.
    ##idx_train = [68,  7, 75]
    for i, idx in enumerate(idx_test):
        data, mask, _, _, _, _ = vl.val_loader(idx=idx, is_train=True)
        pet = data
        pet = pet.numpy()
        print(pet.shape)
        print(pet.sum())
        pets.append(pet)
        mask = mask.numpy()
        masks.append(mask)
    return pets, masks

if __name__ == "__main__":
 
 
    pets, masks = get_params()
    for i in range(len(masks)):
        
        print('sum:',pets[i].sum())
