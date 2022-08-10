import torch
import nrrd
import nibabel as nib
from utils.Utils import cut_pet
from data_augmentation import *

## Data Generator

## Data Loader

class DataLoader():
    def __init__(self, list_IDs, directory, dtype, norm,
                 dim=(64, 64, 64), n_channels=1, train=True, augmentation=True):
        self.dim = dim
        self.directory = directory
        self.list_IDs = list_IDs
        self.dtype = dtype
        self.norm = norm
        self.n_channels = n_channels
        self.train = train
        self.augmentation = augmentation

    def Loading(self):

        if self.train:
            X, y = self.Data_loading()
            # printing(X, y)
            return X, y
        else:
            X, _ = self.Data_loading()
            return X

    def Data_loading(self):
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((len(self.list_IDs), self.n_channels, *self.dim), dtype=np.float32)
        y = np.empty((len(self.list_IDs), 1, *self.dim), np.float32)
        
        if self.dtype == 'nrrd':
            image_loader = nrrd_loader
        elif self.dtype == 'nifti':
            image_loader = nifti_loader
        else:
            return NotImplementedError('loader not implemented for this image data type')
        
        # Generate data
        for i, ID in enumerate(self.list_IDs):
            # Store cut pet
            # get the index of the patient in the file list
            p_index = self.list_IDs.index(ID)
            # print(p_index)
            pet, _ = image_loader(self.directory + '/' + ID + '/PET.' + self.dtype)
            
            contour, _ = image_loader(self.directory + '/' + ID + '/prostate.' + self.dtype)

            # set everything outside the prostate to zero
            x_cut, prost_cut  = cut_pet(pet, contour, normalization=self.norm,concat=True)
            if self.n_channels ==1:
                x_cut = x_cut
            elif self.n_channels > 1:
                x_cut = np.expand_dims(x_cut,axis=0)
                prost_cut = np.expand_dims(prost_cut,axis=0)
                x_cut = np.concatenate((x_cut,prost_cut),axis=0)
                
            if self.train:
                # Store cut lesion
                lesion, _ = image_loader(self.directory + '/' + ID + '/l1.' + self.dtype)
                #lesion = np.multiply(lesion, contour)
                y_cut = cut_pet(lesion, contour, normalization='none')
                y_cut = np.where(y_cut > 0, 1.0, 0)
                y_cut = np.where(prost_cut == 1.0, y_cut, 0.0)
                
                ## Data_Augmentation
                # to do
                if self.augmentation:
                    print('Performing Augmentation.....')
                    x_cut, y_cut = Data_Augmentation(x_cut, y_cut, index=p_index, rotation_angle=30,
                                                     width_shift_range=5,
                                                     height_shift_range=5, horizontal_flip=True, vertical_flip=False,
                                                     fill_mode='nearest', order=3)

                X[i,] = x_cut
                y[i,] = y_cut


            else:

                X[i,] = torch.tensor(x_cut)

        return X, y


class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, x_train, mask_train):
        'Initialization'
        self.x_train = x_train
        self.mask_train = mask_train

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.x_train)

  def __getitem__(self, index):
        'Generates one sample of data'

        # Load data and get label
        X = self.x_train[index]
        print(X.shape)
        y = self.mask_train[index]

        return X, y

# image loader for nifti images
def nifty_loader(image_dir):
    image = nib.load(image_dir)
    header = image.header
    data = image.get_fdata()
    return data, header

# image loader for nrrd image
def nrrd_loader(image_dir):
    image, header = nrrd.read(image_dir)
    return image, header
