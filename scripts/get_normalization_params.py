"""
Copyright Dejan Kostyszyn 2019

This method computes the standard deviation and mean of the
complete dataset. This is then used for normalizing the data
for preprocessing.

mean = x.sum() / N, where N = x.size
std = sqrt(mean(|x - x.mean()|**2))
"""

import ReadData as rd
import numpy as np
import torch, nrrd, time, csv, utils
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

    # If you want a manual training permutation enter it here.
    # idx_train = [72,60,132,71,31,101,144,158,96,166,164,161,125,56,124,93,121,70,45,57,122,29,77,49,90,82,48,159,66,128,38,150,58,171,44,163,120,22,21,30,137,33,52,138,140,102,80,106,94,110,105,34,97,88,160,41,165,154,111,113,67,89,61,115,112,26,63,119,149,155,157,73,51,130,116,174,156,133,143,98,95,39,131,50,126,100,85,59,43,167,135,134,141,142,74,91,87,68,145,78,170,108,25,62,103,153,69,76,40,139,129,117,81,86,84,65,118,92,55,37,99,83,32,0,169,79,19,148,136,35,36,104,23,168,42,173,123,127,162,109,172,54,107,147,146]

    print("Computing mean and std over training data: {}".format(idx_train))

    # Check if results folder exists and if so, ask user if really want to continue.
    results_path = join(opt.results_path)
    utils.overwrite_request(results_path)
    utils.create_folder(results_path)
    filepath = join(results_path) + "mean_std.csv"
    
    # Create file for storing results.
    with open(filepath, "w", newline="") as file:
        writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["standard deviation", "arithmetic mean"])

    # Compute mean.
    for i, idx in enumerate(idx_train):
        data, _, _, _, _, _ = vl.val_loader(idx=idx, is_train=False)
        pet = data[0]
        pet = pet.numpy()

        mean += pet.sum()
        nr_of_elements += pet.size

        print("{:.2f}% completed.".format((i / len(idx_train)) * 100 / 2), end="\r")

    mean /= nr_of_elements

    # Compute standard deviation.
    for i, idx in enumerate(idx_train):
        data, _, _, _, _, _ = vl.val_loader(idx=idx, is_train=False)
        pet = data[0]
        pet = pet.numpy()

        std += np.power(np.abs(pet - mean), 2).sum()

        print("{:.2f}% completed.".format((i / len(idx_train)) * 100 + 50), end="\r")
    print("{:.2f}% completed.".format(100), end="\r")

    std = np.sqrt(std / nr_of_elements)

    # Write results into file.
    with open(filepath, "a", newline="") as file:
        writer = csv.writer(file, delimiter=",", quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([std, mean])
    print("Results written into file {}".format(filepath))

    return std, mean
    
if __name__ == "__main__":
    start_time = time.time()
    std, mean = get_params()
    print("std = {}, mean = {}".format(std, mean))
    print("Completed in {}s".format(time.time() - start_time))
    print("Please insert the values for std and mean into file ReadData.py!")