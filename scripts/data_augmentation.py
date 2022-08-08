import numpy as np
import random
import scipy




# random rotation
def random_rotation(x, y, angle, fill_mode='nearest', cval=0., interpolation_order=3):
    if bool(random.getrandbits(1)):

        theta = random.uniform(- angle, angle)
        # print('angle: ', theta)
        x_rot = scipy.ndimage.interpolation.rotate(x, theta, order=interpolation_order, mode=fill_mode, axes=(0, 1),
                                                   cval=cval, reshape=False)
        y_rot = scipy.ndimage.interpolation.rotate(y, theta, order=interpolation_order, mode=fill_mode, axes=(0, 1),
                                                   cval=cval, reshape=False)
    else:
        x_rot = x
        y_rot = y

    return x_rot, y_rot

# random shift
def random_shift(x, y, x_shift_range, y_shift_range, fill_mode='nearest', cval=0., interpolation_order=3):
    if bool(random.getrandbits(1)):

        # shift by pixels
        x_shift = random.randint(-x_shift_range, x_shift_range)
        y_shift = random.randint(-y_shift_range, y_shift_range)
        # print('height', x_shift)
        # print('width', y_shift)
        shift = np.array([x_shift, y_shift, 0])
        x_shifted = scipy.ndimage.shift(x, shift, output=None, order=interpolation_order, mode=fill_mode, cval=cval,
                                        prefilter=True)
        y_shifted = scipy.ndimage.shift(y, shift, output=None, order=interpolation_order, mode=fill_mode, cval=cval,
                                        prefilter=True)

    else:

        x_shifted = x
        y_shifted = y

    return x_shifted, y_shifted

# flip axis
def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

# random zoom
def random_zoom(x, y, zoom_range, fill_mode='nearest', cval=0., interpolation_order=3):
    if bool(random.getrandbits(1)):

        # shift by pixels
        zoom = random.uniform(1, zoom_range)

        zoom = np.array([zoom, zoom, 1])
        x_zoomed = scipy.ndimage.zoom(x, zoom, output=None, order=interpolation_order, mode=fill_mode, cval=cval,
                                      prefilter=True)
        y_zoomed = scipy.ndimage.zoom(y, zoom, output=None, order=interpolation_order, mode=fill_mode, cval=cval,
                                      prefilter=True)

        # crop image to be 64x64x64

        if x_zoomed.shape[0] % 2 != 0:
            center = round(x_zoomed.shape[0])
        else:
            center = x_zoomed.shape[0]
        min_v = int(center / 2 - x.shape[0] / 2)
        max_v = int(center / 2 + x.shape[0] / 2)
        x_zoomed = x_zoomed[min_v:max_v, min_v:max_v, :]
        y_zoomed = y_zoomed[min_v:max_v, min_v:max_v, :]

    else:
        x_zoomed = x
        y_zoomed = y

    return x_zoomed, y_zoomed


# Data augmentation function
def Data_Augmentation(x, y, index, fill_mode = 'nearest', order = 3, cval = 0.,
                      rotation_angle = None, width_shift_range = None,
                      height_shift_range = None, zoom = None, horizontal_flip = False, vertical_flip =False):

    x_size = x.shape
    y_size = y.shape
    x = np.squeeze(x)
    if y.ndim == 4:
        y = np.squeeze(y)

    # perform rotation along the x and y axes
    if rotation_angle is not None:
        random.seed(index)
        x, y = random_rotation(x, y, rotation_angle, fill_mode= fill_mode,
                               cval=cval, interpolation_order = order)

    # perform shift along the x and y axes
    # width_shift_range for y
    # height_shift_range for x

    if width_shift_range is not None or height_shift_range is not None:
        if not isinstance(height_shift_range, int) or not isinstance(width_shift_range, int):
            raise ValueError(
                'Shift range values should be integers.')

        random.seed(index + 1)
        x, y = random_shift(x ,y, x_shift_range = height_shift_range, y_shift_range = width_shift_range,
                            fill_mode= fill_mode,
                            cval=cval, interpolation_order = order)

    if  horizontal_flip:
        random.seed(index + 2)
        if bool(random.getrandbits(1)):
            img_col_axis = 1
            x = flip_axis(x, img_col_axis)
            y = flip_axis(y, img_col_axis)
            # print('flip')

    if  vertical_flip:
        random.seed(index + 2)
        if bool(random.getrandbits(1)):
            img_row_axis = 0
            x = flip_axis(x, img_row_axis)
            y = flip_axis(y, img_row_axis)

    if zoom is not None:
        random.seed(index + 3)
        x, y = random_zoom(x, y, zoom, fill_mode= fill_mode,
                           cval=cval, interpolation_order = order)

    x = x.reshape(x_size)
    y = y.reshape(y_size)

    y =  np.uint8(y)

    return x, y
