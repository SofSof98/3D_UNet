import numpy as np
import matplotlib.pyplot as plt
import os, torch, medpy, sys, torch, random
from medpy.metric.binary import hd, dc, asd, assd, precision,\
    sensitivity, specificity

def str_to_bool(value):
    """
    Turns a string into boolean value.
    """
    t = ['true', 't', '1', 'y', 'yes', 'ja', 'j']
    f = ['false', 'f', '0', 'n', 'no', 'nein']
    if value.lower() in t:
        return True
    elif value.lower() in f:
        return False
    else:
        raise ValueError("{} is not a valid boolean value. Please use one out of {}".format(value, t + f))

def divide_data(max_idx, opt):
    """
    This method shuffles the data and then divides it into
    80% for training
    10% for validation
    10% for testing
    """
    idx = np.arange(max_idx)

    # Planting a seed for reproducibility.
    np.random.seed(2020)
    torch.manual_seed(2020)
    if torch.cuda.is_available():
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False

    if opt.shuffle == True:
        np.random.shuffle(idx)
        
    return idx[:int(len(idx)*0.7)], idx[int(len(idx)*0.7):int(len(idx)*0.8)], idx[int(len(idx)*0.8):]

def fit_pet_shape(pet, prostate_contour):
    # Cut data, restricted to the prostate contours + a pitch per direction per dimension.
    """
    nrrd has the following format, assuming to watch the patient from the front:
    (x, y, z)
    x: left to right (ascending)
    y: front to back (ascending)
    z: bottom to top (ascending)
    """
    pitch = 5
    pattern = np.where(prostate_contour == 1)

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
    limit = 64

    while maxx - minx < limit:
        maxx += 1
        minx -= 1

    while maxy - miny < limit:
        maxy += 1
        miny -= 1
        
    while maxz - minz < limit:
        maxz += 1
        minz -= 1

    pet = torch.FloatTensor(pet)
    pet = pet[minx:maxx,miny:maxy,minz:maxz]
    prostate_contour = torch.FloatTensor(prostate_contour)
    prostate_contour = prostate_contour[minx:maxx,miny:maxy,minz:maxz]

    return pet, prostate_contour, minx, maxx, miny, maxy, minz, maxz

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

def set_optimizer(opt, model_params):
    if opt.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model_params, lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps)
    elif opt.optimizer.lower() == "adamax":
        optimizer = torch.optim.Adamax(model_params, lr=0.002, betas=(0.9, 0.999), eps=opt.eps, weight_decay=opt.weight_decay)
    else:
        sys.exit("{} is not a valid optimizer. Choose one of {adam, adamax}".format(opt.optimizer))
    return optimizer

# Logic or.
def tensor_or(t1, t2):
    return (t1 + t2) >= 1.0

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


def is_valid_gs(GS):
    """
    Checks whether the gleason score (GS) of a cancer lesion is valid or not.
    Valid scores are:
    0: GS <= 7
    1: GS > 7
    """
    if int(GS) in [0, 1]:
        return True
    else:
        return False

def check_data_consistency(patient_id, *args):
    """
    Checks whether the data is consistent or not. Expects:
    patient_id: The ID of the patient, the data has to be checked of
    headder 1: type:OrderedDict or type:dict headder of a nrrd file
    headder 2: type:OrderedDict or type:dict headder of a nrrd file
    ...
    The function can handle multiple headder files, but minimal 2.
    Checks values for dictionnary keywords:
    - type
    - dimension
    - space
    - sizes
    - space directions
    - kinds
    - endian
    - encoding
    - space origin
    """
    if len(args) < 2 or not isinstance(patient_id, str):
        sys.exit("Use: check_data_consistency(<str>, <OrderedDict>, <OrderedDict>, ...)")

    dimension = args[0]["dimension"]
    space = args[0]["space"]
    sizes = args[0]["sizes"]
    space_directions = args[0]["space directions"]
    kinds = args[0]["kinds"]
    encoding = args[0]["encoding"]
    space_origin = args[0]["space origin"]

    for i in range(1,len(args)):
        headder = args[i]
        if dimension != headder["dimension"]:
            sys.exit("Data of {} is not consistent. Please check the dimensions: {} != {}.".format(
                patient_id, dimension, headder["dimension"]
            ))
        elif space != headder["space"]:
            sys.exit("Data of {} is not consistent. Please check the space: {} != {}.".format(
                patient_id, space, headder["space"]
            ))
        elif not np.array_equal(np.array(sizes), np.array(headder["sizes"])):
            sys.exit("Data of {} is not consistent. Please check the sizes: {} != {}.".format(
                patient_id, sizes, headder["sizes"]
            ))
        elif not np.array_equal(np.array(space_directions), np.array(headder["space directions"])):
            sys.exit("Data of {} is not consistent. Please check the space directions: {} != {}.".format(
                patient_id, space_directions, headder["space directions"]
            ))
        elif kinds != headder["kinds"]:
            sys.exit("Data of {} is not consistent. Please check the kinds: {} != {}.".format(
                patient_id, kinds, headder["kinds"]
            ))
        elif encoding != headder["encoding"]:
            sys.exit("Data of {} is not consistent. Please check the encoding: {} != {}.".format(
                patient_id, encoding, headder["encoding"]
            ))
        elif not np.array_equal(np.array(space_origin), np.array(headder["space origin"])):
            sys.exit("Data of {} is not consistent. Please check the space origin: {} != {}.".format(
                patient_id, space_origin, headder["space origin"]
            ))

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Created folder(s) {}".format(path))
    else:
        print("Folder(s) {} already exist(s).".format(path))

#only for 2 labels
def dice_coeff(y_pred, y_true):
    smooth = 0.000001

    # Flatten
    y_true_f = torch.reshape(y_true, (-1,))
    y_pred_f = torch.reshape(y_pred, (-1,))
    intersection = torch.sum(y_true_f * y_pred_f)
    score = 2. * intersection / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score

def dice_loss(y_pred, y_true):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    zero = torch.Tensor([0]).to(device)
    one = torch.Tensor([1]).to(device)
    losses = []

    dims = range(y_pred.shape[0])

    for i in dims:
        losses.append(1 - dice_coeff(y_pred[i,...], torch.where(y_true == i, one, zero)))

    loss = losses[0]
    for i in range(1, dims[-1]):
        loss += losses[i]

    return loss



def Tversky_coeff(y_pred, y_true):
    alpha = 0.7
    beta = 0.3
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

def tv_loss(y_pred, y_true):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    zero = torch.Tensor([0]).to(device)
    one = torch.Tensor([1]).to(device)
    losses = []

    dims = range(y_pred.shape[0])

    for i in dims:
        losses.append(1 - Tversky_coeff(y_pred[i,...], torch.where(y_true == i, one, zero)))

    loss = losses[0]
    for i in range(1, dims[-1]):
        loss += losses[i]

    return loss

def focal_tv_loss(y_pred, y_true):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    zero = torch.Tensor([0]).to(device)
    one = torch.Tensor([1]).to(device)
    losses = []
    gamma = 4/3

    dims = range(y_pred.shape[0])

    for i in dims:
        losses.append((1 - Tversky_coeff(y_pred[i,...])**gamma, torch.where(y_true == i, one, zero)))

    loss = losses[0]
    for i in range(1, dims[-1]):
        loss += losses[i]

    return loss

def set_loss_fn(opt, num_classes, device):
    if opt.loss_fn.lower() not in ["weighted_cross_entropy", "cross_entropy", "dice", "l2", "l1"]:
        sys.exit('Value for loss_fn must be either of {"weighted_cross_entropy", "cross_entropy", "dice", "l2", "l1"}')
    if opt.loss_fn == "cross_entropy":
        loss_fn = torch.nn.CrossEntropyLoss()
    elif opt.loss_fn == "weighted_cross_entropy":
        tmp = opt.weights
        if tmp == None:
            sys.exit("Please use the --weights flag to define weights for the cross entropy loss.")
        if num_classes == 2:
            print(num_classes)
            print(tmp)
            if not len(tmp) == 2:
                sys.exit("Cross entropy loss needs one weight for each of the two classes. E.g.: [1, 25]")
            weights = torch.FloatTensor(tmp)
            weights = weights.to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weights)
        else:
            print(num_classes)
            print(tmp)
            if not len(tmp) == 3:
                sys.exit("Cross entropy loss needs one weight for each of the three classes. E.g.: [1, 25, 25]")
            weights = torch.FloatTensor(tmp)
            weights = weights.to(device)
            loss_fn = torch.nn.CrossEntropyLoss(weights)
    elif opt.loss_fn == "dice":
        loss_fn = dice_loss
    elif opt.loss_fn.lower() in ["l1", "l2"]:
        loss_fn = torch.nn.MSELoss()
    elif opt.loss_fn.lower() == "l1":
        loss_fn = torch.nn.L1Loss()
    elif opt.loss_fn == "tv":
        loss_fn = tv_loss
    elif opt.loss_fn == "focal_tv":
        loss_fn == "focal_tv_loss"
    return loss_fn

def overwrite_request(path):
    if os.path.exists(path):
        valid = False
        while not valid:
            answer = input("{} already exists. Are you sure you want to overwrite everything in this folder? [yes/no]\n".format(path))
            if str_to_bool(answer):
                valid = True
            elif not str_to_bool(answer):
                sys.exit(1)

def random_flip(data, lesion):
    if bool(random.getrandbits(1)):
        data = data.flip(dims=(1,))
        lesion = lesion.flip(dims=(1,))
    return data, lesion

def random_scaling(opt, data, lesion):
    old_shape = data[0].shape
    final_shape = 64
    interpolation_mode = opt.interpolation_mode

    # Computing random scale factor.
    max_scale_factor = data.shape[1] / final_shape
    
    scale_factor_x = np.random.uniform(1 - ((1-max_scale_factor)/2), 1 + ((1-max_scale_factor)/2))
    scale_factor_y = np.random.uniform(1 - ((1-max_scale_factor)/2), 1 + ((1-max_scale_factor)/2))
    scale_factor_z = np.random.uniform(1 - ((1-max_scale_factor)/2), 1 + ((1-max_scale_factor)/2))

    # Scaling.
    pet = torch.nn.functional.interpolate(data[0].unsqueeze(0).unsqueeze(0),\
        scale_factor=(scale_factor_x, scale_factor_y, scale_factor_z), mode=interpolation_mode).squeeze().squeeze()
    prostate_contour = torch.nn.functional.interpolate(data[1].unsqueeze(0).unsqueeze(0),\
        scale_factor=(scale_factor_x, scale_factor_y, scale_factor_z), mode=interpolation_mode).squeeze().squeeze()
    lesion = torch.nn.functional.interpolate(lesion.unsqueeze(0).unsqueeze(0),\
        scale_factor=(scale_factor_x, scale_factor_y, scale_factor_z), mode=interpolation_mode).squeeze().squeeze()

    # Cropping into old shape.
    minx = 0
    maxx = pet.shape[0]
    miny = 0
    maxy = pet.shape[1]
    minz = 0
    maxz = pet.shape[2]

    if old_shape[0] > final_shape:
        if (maxx - minx) % 2 != 0:
            maxx -= 1
        if (maxy - miny) % 2 != 0:
            maxy -= 1
        if (maxz - minz) % 2 != 0:
            maxz -= 1
    else:
        if (maxx - minx) % 2 != 0:
            maxx += 1
        if (maxy - miny) % 2 != 0:
            maxy += 1
        if (maxz - minz) % 2 != 0:
            maxz += 1

    """
    Choose all tensors to have size of 64x64x64
    """
    while maxx - minx > final_shape:
        maxx -= 1
        minx += 1

    while maxy - miny > final_shape:
        maxy -= 1
        miny += 1
        
    while maxz - minz > final_shape:
        maxz -= 1
        minz += 1

    pet = pet[minx:maxx,miny:maxy,minz:maxz]
    prostate_contour = prostate_contour[minx:maxx,miny:maxy,minz:maxz]
    data = torch.cat((pet.unsqueeze(0), prostate_contour.unsqueeze(0)), dim=0)
    lesion = lesion[minx:maxx,miny:maxy,minz:maxz]

    # Correct the lesion to have only int values.
    lesion = np.where(lesion < 0.5, 0, lesion)
    lesion = np.where(np.logical_and(0.5 <= lesion, lesion < 1.5), 1, lesion)
    lesion = np.where(1.5 <= lesion, 2, lesion)
    lesion = torch.tensor(lesion)
    return data, lesion

def data_augmentation(opt, data, lesion):
    if opt.augmentation.lower() == "none":
        return data, lesion
    elif opt.augmentation.lower() == "flip":
        return random_flip(data, lesion)
    elif opt.augmentation.lower() == "scaling":
        return random_scaling(opt, data, lesion)
    elif opt.augmentation.lower() == "all":
        data, lesion = random_flip(data, lesion)
        return random_scaling(opt, data, lesion)        

        """
        A test if the interpolation produces values that are not in {0,1}
        
        zeros = np.count_nonzero(lesion == 0)
        ones = np.count_nonzero(lesion == 1)
        twos = np.count_nonzero(lesion == 2)
        lgth = (lesion.shape[0]*lesion.shape[1]*lesion.shape[2])

        print("zeros: {}".format(zeros))
        print("ones: {}".format(ones))
        print("twos: {}".format(twos))
        print("all together: {}".format(lgth))
        print("others: {}".format(lgth-(zeros + ones)))

        if (zeros + ones + twos) != lgth:
            print("There are numbers that are neither 0 nor 1!!!!")
        else:
            print("All values are valid.")
        """
    else:
        sys.exit("Use one value out of the set {None, flip, all}")

def rescale(data, lesion):
    PET = data[0]
    final_shape = 64
    minx = 0
    maxx = PET.shape[0]
    miny = 0
    maxy = PET.shape[1]
    minz = 0
    maxz = PET.shape[2]

    if (maxx - minx) % 2 != 0:
        maxx -= 1
    if (maxy - miny) % 2 != 0:
        maxy -= 1
    if (maxz - minz) % 2 != 0:
        maxz -= 1

    """
    Choose all tensors to have size of 64x64x64
    """
    while maxx - minx > final_shape:
        maxx -= 1
        minx += 1

    while maxy - miny > final_shape:
        maxy -= 1
        miny += 1
        
    while maxz - minz > final_shape:
        maxz -= 1
        minz += 1

    PET = PET[minx:maxx,miny:maxy,minz:maxz]
    prostate_contour = data[1]
    prostate_contour = prostate_contour[minx:maxx,miny:maxy,minz:maxz]
    data = torch.cat((PET.unsqueeze(0), prostate_contour.unsqueeze(0)), dim=0)
    lesion = lesion[minx:maxx,miny:maxy,minz:maxz]

    return data, lesion

def fit_prediction_to_pet(pet_shape, pred, minx, maxx, miny, maxy, minz, maxz, opt):
    t = np.full(pet_shape, 0, dtype=float)

    t[minx:maxx, miny:maxy, minz:maxz] = pred

    return t
