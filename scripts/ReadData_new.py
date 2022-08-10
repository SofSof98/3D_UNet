import numpy as np
import nrrd
import torch
import os
import re
import time
import csv
import sys
from natsort import natsorted, ns
from openpyxl import Workbook, load_workbook
import util
from os.path import join

"""
Returns:
PSMA/PET file paths as python3 list,
prostate contour file paths as python3 list,
Cancer infected segments of PET paths as python3 list,
Gleason Score as pytorch float tensor

If is_histo is set to True, the function returns only histo files,
otherwise it returns all files belonging to the expert's predictions.
"""

"""
Loads the PET, cancer segment and prostate contour values and
returns them as pytorch float tensors. It only returns one tensor
each, consisting out of one patients data.
"""


class ValLoader():
    def __init__(self, opt):

        self.opt = opt
        self.data_root = opt.data_root

        self.pet_file_paths,\
            self.cancer_lesionment_paths,\
            self.p_id,\
            self.prostate_contour_paths = self.read_data_paths()

        self.pet_header = None
        self.prostate_contour_headder = None
        self.pet_original_shape = None
        self.lesion_original_shape = None
        self.prostate_contour_original_shape = None

        self.loadedFiles = {}
        # This variable is set, so that the main memory does not explode.
        if "store_loaded_data" in self.opt:
            if not self.opt.store_loaded_data:
                self.max_nr_of_files_to_store = 0
            else:
                self.max_nr_of_files_to_store = 150

        ################################################################
        ################################################################
        ################### INSERT STD AND MEAN HERE ###################
        # Standard deviation to manually change.
        self.std = 6.211471196036908
        print(self.std)
        # Arithmetic mean to manually change.
        self.mean = 1.8105808395450398
        ################################################################
        ################################################################

    def read_data_paths(self):
        patient_ids = dict()
        pet_file_paths = dict()  # key: patient folder, value: name of PET file
        # key: patient folder, value: name of prostate contour file
        prostate_contour_paths = dict()
        # key: patient folder, value: List(Tuple(name of lesion file, GS))
        cancer_lesionment_paths = dict()

        # Loading CSV file
        filename = self.opt.csv_file
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            try:
                i = 0
                for row in reader:
                    if row[0] not in patient_ids.values():
                        patient_ids.setdefault(i, row[0])
                        i += 1
                    pet_file_paths.setdefault(
                        row[0], join(self.data_root, row[0], row[1]))
                    prostate_contour_paths.setdefault(
                        row[0], join(self.data_root, row[0], row[2]))
                    cancer_lesionment_paths.setdefault(row[0], []).append(
                        (join(self.data_root, row[0], row[3]), row[4]))
            except csv.Error as e:
                sys.exit('file %s, line %d: %s' %
                         (filename, reader.line_num, e))

        return pet_file_paths, cancer_lesionment_paths, patient_ids, prostate_contour_paths

    def nr_of_patients(self):
        return len(self.p_id)

    # Logic or.
    def tensor_or(self, t1, t2):
        return (t1 + t2) >= 1.0

    # Combining cancer leasons and depending gleason scores in one tensor.
    """
    class 0: no lesion
    class 1: gleason score <= 7
    class 2: gleason score > 7
    """

    def combine_lesion_and_gs(self, lesion1, gs1, lesion2, gs2):
        t1 = lesion1.numpy()
        t2 = lesion2.numpy()
        t = np.full(t1.shape, 2, dtype=int)
        t = np.where(t1 == 1, gs1, t)
        t = np.where(t2 == 1, gs2, t)
        return torch.tensor(t)

    def val_loader(self, idx=0, is_train=True):
        """
          params:
            idx (int): id of patient who's data shall be loaded.
            is_train (bool): load training or validation data. This is used in cases where data loading differs between training and validation procedure.
        """
        # Assigning index to person ID.
        p_id = self.p_id[idx]

        # return already loaded data.
        if p_id in self.loadedFiles and (self.opt.augmentation.lower() not in ["all", "scaling", "flip"] or is_train==False):
            return self.loadedFiles[p_id]

        # Reading PET file and converting it to pytorch float tensor.
        readdata, self.pet_header = nrrd.read(self.pet_file_paths[p_id])
        readdata = np.float32(readdata)
        pet = torch.FloatTensor(readdata)

        # Putting all cancer lesions into one pytorch tensor.
        val = self.cancer_lesionment_paths[p_id]
        
        readdata, lesion_headder = nrrd.read(val[0][0])
        lesion_part = np.float32(readdata)
        cancer_lesion = np.full(lesion_part.shape, 0, dtype=int)
        cancer_lesion = np.where(lesion_part > 0, 1, cancer_lesion)

        x = 1
        while x < len(val):
            readdata, lesion_headder2 = nrrd.read(val[x][0])
            lesion_part = np.float32(readdata)

            # Check data consistency
            if self.opt.check_consistency == True:
                utils.check_data_consistency(
                    p_id, lesion_headder, lesion_headder2)

            cancer_lesion = np.where(lesion_part > 0, 1, cancer_lesion)
            x += 1
        cancer_lesion = torch.FloatTensor(cancer_lesion.astype(float))

        # Putting prostate contour in pytorch tensor.
        readdata, self.prostate_contour_headder = nrrd.read(
            self.prostate_contour_paths[p_id])

        readdata = np.float32(readdata)
        prostate_contour = torch.FloatTensor(readdata)

        # Check consistency of data..
        if self.opt.check_consistency == True:
            utils.check_data_consistency(
                p_id, self.pet_header, self.prostate_contour_headder, lesion_headder)

        # Cut data, restricted to the prostate contours + a pitch per direction per dimension.
        """
        nrrd has the following format, assuming to watch the patient from the front:
        (x, y, z)
        x: left to right (ascending)
        y: front to back (ascending)
        z: bottom to top (ascending)

        The pitch will increase the volume around the prostate contour that will be used.
        This is done, because cancer lesions can lie outside the prostate contour volume.
        """

        pitch = 5
        cont = prostate_contour.numpy()
        pattern = np.where(cont == 1)

        minx = np.min(pattern[0]) - pitch
        maxx = np.max(pattern[0]) + pitch
        miny = np.min(pattern[1]) - pitch
        maxy = np.max(pattern[1]) + pitch
        minz = np.min(pattern[2]) - pitch
        maxz = np.max(pattern[2]) + pitch

        if (maxx - minx) % 2 != 0:
            maxx += 1
        if (maxy - miny) % 2 != 0:
            maxy += 1
        if (maxz - minz) % 2 != 0:
            maxz += 1

        """
        Choose all tensors to have size of 64x64x64
        """
        if self.opt.augmentation.lower() not in ["all", "scale"] or is_train==False:
            limit = 64
        else:
            # Past: 75
            limit = int(64 * (1+self.opt.augmentation_radius/100/2))

        while maxx - minx < limit:
            maxx += 1
            if minx == 0:
                maxx += 1
            else:
                minx -= 1

        while maxy - miny < limit:
            maxy += 1
            if miny == 0:
                maxy += 1
            else:
                miny -= 1

        while maxz - minz < limit:
            maxz += 1
            if minz == 0:
                maxz += 1
            else:
                minz -= 1

        old_pet_shape = (pet.shape, minx, maxx, miny, maxy, minz, maxz)
        pet = pet[minx:maxx, miny:maxy, minz:maxz].float()

        # Zero-mean normalization
        if self.opt.normalize.lower() == "global":
            pet = pet - self.mean
            pet = pet / self.std
        elif self.opt.normalize.lower() == "local":
            pet = (pet - pet.min()) / (pet.max() - pet.min())
        elif self.opt.normalize.lower() != "none":
            sys.exit("{} is not a valid value for normalize option. Choose one out of the set {None, local, global}".format(
                self.opt.normalize))

        cancer_lesion = cancer_lesion[minx:maxx, miny:maxy, minz:maxz].float()
        prostate_contour = prostate_contour[minx:maxx, miny:maxy, minz:maxz].float()

        # Compute the ratio between cancer affected regions and non affected regions.
        # TODO: Refactor
        if self.opt.return_ratio == True:
            twos = np.count_nonzero(cancer_lesion == 2)
            ones = np.count_nonzero(cancer_lesion == 1)
            zeros = np.count_nonzero(cancer_lesion == 0)
            ratio = np.array([zeros, ones, twos])
            print("ratio = {}".format(ratio))

        zeros = np.count_nonzero(cancer_lesion == 0)
        ones = np.count_nonzero(cancer_lesion == 1)

        ratio = np.array([zeros, ones])

        # Cut cancer lesions depending on prostate contour.
        cancer_lesion = torch.where(prostate_contour == 1., cancer_lesion, torch.tensor([0.]))

        # Data augmentation.
        if is_train == True:
          if self.opt.augmentation.lower() in ["all", "scaling"]:
              new_shape = (64, 64, 64)
              interpolation_mode = "trilinear"
              pet = torch.nn.functional.interpolate(pet.unsqueeze(0).unsqueeze(0),\
                size=new_shape, mode=interpolation_mode).squeeze().squeeze()
              prostate_contour = torch.nn.functional.interpolate(prostate_contour.unsqueeze(0).unsqueeze(0),\
                size=new_shape, mode=interpolation_mode).squeeze().squeeze()
              cancer_lesion = torch.nn.functional.interpolate(cancer_lesion.unsqueeze(0).unsqueeze(0),\
                size=new_shape, mode=interpolation_mode).squeeze().squeeze()

              # Make prostate contour and cancer lesion binary again.
              prostate_contour = torch.where(prostate_contour < 0.5, torch.tensor(0.0), torch.tensor(1.0)).float()
              cancer_lesion = torch.where(cancer_lesion < 0.5, torch.tensor(0.0), torch.tensor(1.0))

          if self.opt.augmentation.lower() in ["flip", "all"]:
              if bool(random.getrandbits(1)):
                  pet = pet.flip(dims=(1,))
                  prostate_contour = prostate_contour.flip(dims=(1,))
                  cancer_lesion = cancer_lesion.flip(dims=(1,))

        # Concatenate PET and and prostate contour.
        #pet =  torch.where(prostate_contour == 1., pet, torch.tensor([0.]))
        data = torch.cat((pet.unsqueeze(0), prostate_contour.unsqueeze(0)), dim=0)

        if self.max_nr_of_files_to_store > 0 and (self.opt.augmentation.lower() not in ["all", "scaling", "flip"]) or (is_train==False):
            self.loadedFiles[p_id] = [data, cancer_lesion,
                                    p_id, self.pet_header, ratio, old_pet_shape]
            self.max_nr_of_files_to_store -= 1
        return data, cancer_lesion, p_id, self.pet_header, ratio, old_pet_shape
