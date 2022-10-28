import argparse



def get_parser():
    # parameter priority: command line > config > default

    parser = argparse.ArgumentParser(
        description='Intraprostatic Lesion Segmentation')
    parser.add_argument('--data_dir', type=str, default='True', required=True, help='Data Dir.')
    parser.add_argument('--txt_dir', type=str, default='True', required=True, help='Image List Dir')
    parser.add_argument('--results_dir', default='True', required=True, help='Results Dir')
    parser.add_argument('--augmentation', type=arg_boolean, default=False, required=False, help='Perform Data Augmentation')
    parser.add_argument('--normalize', type=str, default='global', required=False, help='Choose type of normalization {None, local, global}.')
    parser.add_argument('--image_type', type=str, default='nrrd', required=False, help='Image data type {nrrd, nifti}')
    parser.add_argument('--scheduling_lr', type=str, default=None, help = 'Schedule learning rate during training {None, LRScheduler, ReduceLROnPlateau}')
    parser.add_argument('--save_freq', type=int, default=100, required=False, help='Saved checkpoints freq')
    parser.add_argument('--batch_size', type=int, default=1,required=False, help='training batch size')
    parser.add_argument('--config',default='config/config.yaml',required=False, help='path to the configuration file. If None, default settings will be used')
    parser.add_argument('--test_batch_size', type=int, default=1, help='test batch size')
    parser.add_argument('--loss', type=str, default='dice', required=False, help='Choose loss function: {"binary_cross_entropy", "dice", "dice_bce", "iou","tversky"}')
    parser.add_argument('--dim', type=int, default=[64,64,64], required=False, help='Dimensions input images')
    parser.add_argument('--model', type=str, default='models.Unet', required=False, help='Name of the model to train or test')
    parser.add_argument('--best_model', type=str, default='loss', required=False, help='Name for the trained model to test: {"loss", "dice", "ratio"}.')
    parser.add_argument('--ckpt', type=str, default='ckpt/best_loss_model.ckpt', required=False, help='Ignore.')
    parser.add_argument('--filter_scale', type=int, default=1, required=False, help='Value used to scale the number of filters for each layer in the network')
    parser.add_argument('--n_channels', type=int, default=2, required=False, help='number of input channels')
    parser.add_argument('--batchnorm', type=arg_boolean, default=False, help='normalize layer output')
    parser.add_argument('--norm_type', type=str, default='group', required=False, help='Type of normalization:{"group", "batch"})
    parser.add_argument('--n_groups', type=int, default='16', required=False, help='Number of groups for Group Normalization, n_groups = 1 (layer)')
    parser.add_argument('--shuffle', action='store_true', required=False, help='Shuffle before splitting?')
     # optim
    parser.add_argument('--epochs', type=int, default=1000, required=False, help='Number of training epochs')
    parser.add_argument('--start_epoch', type=int, default=0, required=False, help='Epoch to start training from') 
    parser.add_argument('--optimizer', type=str, default='adam', required=False, help='Choose optimizer {adam, adamax,adagrad, SGD}')
    parser.add_argument('--weight_decay', type=float, default=0., required=False, help='Weight decay for optimizer.')
   
    # parser.add_argument('--scheduler', type=float, default=0, help='initial learning rate')
    parser.add_argument('--lr_SDG', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('--momentum_SDG', type=float, default=0.9, help='initial momentum')
    parser.add_argument('--lr_adam', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, required=False, help='Beta1 for Adam solver.')
    parser.add_argument('--beta2', type=float, default=0.999, required=False, help='Beta2 for Adam solver.')
    parser.add_argument('--eps', type=float, default=1e-3, required=False, help='Epsilon for Adam solver.')# try eps = 1e-3
    parser.add_argument('--eps_adagrad', type=float, default=1e-10, required=False, help='Epsilon for Adagrad solver.')
    parser.add_argument('--lr_adagrad', type=float, default=0.01, help='initial learning rate') # default eps for adagrad
    parser.add_argument('--lr_adamax', type=float, default=0.002, help='initial learning rate') 
    parser.add_argument( '--step', type=int, default=250,nargs='+', help='the epoch where optimizer reduce the learning rate')
    parser.add_argument( '--steps', type=int, default=[100, 200],nargs='+', help='the epochs where optimizer reduce the learning rate')
    parser.add_argument('--nesterov', type=arg_boolean, default=False, help='use nesterov or not')
    parser.add_argument('--dr', type=float, default=0.0, help='Dropout rate')
    parser.add_argument('--dv', type=arg_boolean, default=False, help='Perform deep supervision if allowed')

    
    parser.add_argument('--direct_test', type=arg_boolean, default=True, help='Perform test directly when calling main.py')
    parser.add_argument('--rs', type=int, default='2020', required=False, help='set seed')
                        
    # processor
    parser.add_argument('--phase', default='train', help='Ignore')
    parser.add_argument('--training', type=arg_boolean, default=True, help='Ignore')
    parser.add_argument('--model-args',type=dict, default=dict(), help='Ignore')

    
    return parser

def arg_boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError("{} is not a valid boolean value".format(v))

