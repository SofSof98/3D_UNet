import SimpleITK as sitk
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import cv2
import numpy as np
import nrrd
import os
import nibabel as nib
from nibabel.testing import data_path

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


        
def normalize_PET_data(img_dir, img_loader):

    files = os.listdir(img_dir)

    n_data = int(input("Please enter the number of data samples: "))

    if n_data != len(files):
        raise Exception('Check data folder!')

    outputDir = img_dir + '_normalized'
    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)
        
    for file in files:
        image_dir = os.path.join(img_dir, file)
        data, header = img_loader(image_dir)
        header = header.copy()
        data = data / 1000.0

        if img_loader == nifty_loader:
            new_img = nib.nifti1.Nifti1Image(data, None, header=header)
            print(os.path.join(outputDir, file))
            nib.save(new_img, os.path.join(outputDir, file))

        elif img_loader == nrrd_loader:
            print(os.path.join(outputDir, file))
            nrrd.write(file, data, header)

        else:
            raise Exception('No avaibale data format!')
    return outputDir

def resample_image(itk_image, out_spacing=[2.0, 2.0, 2.0], method = 'linear'):
   
    original_spacing = itk_image.GetSpacing()
    original_size = itk_image.GetSize()

    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(itk_image.GetDirection())
    resample.SetOutputOrigin(itk_image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    #resample.SetDefaultPixelValue(itk_image.GetPixelIDValue())
    
    #  The spline order can also be set, though the default of 
    # cubic is appropriate in most cases
    if method == 'nearest':
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
        
    elif method == 'linear':
        resample.SetInterpolator(sitk.sitkLinear)

    elif method == 'bspline':
        resample.SetInterpolator(sitk.sitkBSpline)

    else:
        raise Exception('No avaibale interpolation method')
    
    return resample.Execute(itk_image)

def resample_volume(volume, out_spacing = [2.0, 2.0, 2.0], method = 'linear'):
    #volume = sitk.ReadImage(volume_path, sitk.sitkFloat32) # read and cast to float32
    original_spacing = volume.GetSpacing()
    original_size = volume.GetSize()
    new_size = [int(round(osz*ospc/nspc)) for osz,ospc,nspc in zip(original_size, original_spacing, out_spacing)]
    if method == 'nearest':
        interpolator = sitk.sitkNearestNeighbor
        
    elif method == 'linear':
        interpolator = sitk.sitkLinear 

    elif method == 'bspline':
        interpolator = sitk.sitkBSpline

    else:
        raise Exception('No avaibale interpolation method')
    
    return sitk.Resample(volume, new_size, sitk.Transform(), interpolator,
                         volume.GetOrigin(), out_spacing, volume.GetDirection(), 0,
                         volume.GetPixelID())

def resample(img_dir, out_dir, resampler, out_spacing=[2.0, 2.0, 2.0], method = 'linear',  in_type = 'nrrd', out_type = 'nrrd', split = '.', filename = 'PET'):

    ''' resampler -> - resample_image
                     - resample_volume '''
    files = os.listdir(img_dir)
    for file in files:
        f = os.path.join(img_dir, file)

        if os.path.isfile(f):
            print(f)
    
            reader = sitk.ImageFileReader()
            if in_type  == 'nifti':
                reader.SetImageIO("NiftiImageIO")
            elif in_type  == 'nrrd':
                reader.SetImageIO("NrrdImageIO")
            else:
                raise Exception('No avaibale data format!')
        
            reader.SetFileName(f)
            image = reader.Execute()
            resampled = resampler(image, out_spacing= out_spacing, method=method)
            size = resampled.GetSize()
            print(size)
            name, _ = file.split(split,1)

            outputDir = out_dir + '/' + name.upper() 
            if not os.path.isdir(outputDir):
                os.mkdir(outputDir)
        
            if out_type  == 'nifti':
                output_name = outputDir + '/'+ filename + '.nii'
                sitk.WriteImage(resampled, output_name ,imageIO="NiftiImageIO")

        
            elif out_type  == 'nrrd':
                output_name = outputDir + '/'+ filename +'.nrrd'
                sitk.WriteImage(resampled, output_name ,imageIO="NrrdImageIO")
        
            else:
                raise Exception('No avaibale data format!')
       

def anonymize_data(in_dir, df):

    files = os.listdir(in_dir)
    for file in files:

        id_pat = df.loc[file.upper()]['ID']
        source = os.path.join(in_dir, file.upper())
        dest = os.path.join(in_dir, id_pat.upper())
        os.rename(source, dest)        
