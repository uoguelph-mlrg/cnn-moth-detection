import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from numpy.lib import stride_tricks
import matplotlib as mpl
import socket
import os
import copy


def sliding_window(im, win_height=128, win_width=64, x_stride=1, y_stride=1):
    """Returns a view win into im where win[i,j] is a view of the
i,j'th window in im."""

    H, W = im.shape[:2]
    # nh = (H - win_height + 1 + y_stride - 1) / y_stride
    # nw = (W - win_width + 1 + x_stride - 1) / x_stride
    nh = (H - win_height + y_stride) / y_stride
    nw = (W - win_width + x_stride) / x_stride

    # Get the original strides.
    strides = np.asarray(im.strides)
    strides_scaled = copy.deepcopy(strides)
    
    # multiply the input needed stride
    strides_scaled[0] = strides[0] * y_stride
    strides_scaled[1] = strides[1] * x_stride
    
     
    # The first two strides also advance in the x,y directions
    new_strides = tuple(np.concatenate((strides_scaled[:2], strides[:2])))
    # print new_strides
    

    # The new shape, this should allow for grayscale/color images in
    # the final position.
    new_shape = tuple([nh, nw, win_height, win_width] + list(im.shape[2:]))

    if len(new_strides) < len(new_shape):
        # this means the input im is color image
        # need to add 1 at the end of new_shape
        
        new_strides = tuple(list(new_strides) + [1])

    # Create a view into the image array.
    windows = stride_tricks.as_strided(im, new_shape, new_strides)

    return windows
# end def sliding_window


def nms(boxes, overlap):
    """ Non-maximal suppression
    Based on code from Tomasz Malisiewicz
    http://quantombone.blogspot.ca/2011/08/blazing-fast-nmsm-from-exemplar-svm.html

    Greedily select high-scoring detections and skip detections that
    are significantly covered by a previously selected detection.
    NOTE: This is adapted from Pedro Felzenszwalb's version (nms.m),
    but an inner loop has been eliminated to significantly speed it up
    in the case of a large number of boxes

    boxes is a n_boxes x 5 array, where the columns are:
    x_upper_left, y_upper_left, x_bottom_right, y_bottom_right, confidence

    The dimensions are Python-style, such that x_bottom_right, y_bottom_right are just outside the bounding box, and
    x_upper_left - x_bottom_right = width
    y_upper_left - y_bottom_right = height
    """
    if boxes.shape[0] == 0:
        return boxes, np.empty((0, ), dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, -1]

    area = (x2 - x1) * (y2 - y1)

    I = np.argsort(s)

    pick = np.cast['int32'](s * 0)
    counter = 0

    while len(I) > 0:
        last = len(I)
        i = I[last - 1]
        pick[counter] = i
        counter += 1

        # finds the intersection of each bounding box with lesser score
        # with the current "top score" bounding box
        xx1 = np.fmax(x1[i], x1[I[:last]])
        yy1 = np.fmax(y1[i], y1[I[:last]])
        xx2 = np.fmin(x2[i], x2[I[:last]])
        yy2 = np.fmin(y2[i], y2[I[:last]])

        w = np.fmax(0.0, xx2 - xx1)
        h = np.fmax(0.0, yy2 - yy1)

        o = w * h / area[I[:last]]

        # the current "top score" bounding box as well as those who
        # overlap more than the threshold are removed from the index

        retain = o <= overlap
        retain[-1] = False
        I = I[retain]
    pick = pick[:counter]
    top = boxes[pick]

    return top, pick
# end def nms


def get_paths():
    '''
    NOTE that this function is not used anywhere currently (140504)
    '''
    if 'islab.soe.uoguelph.ca' in socket.gethostname():
        # we're on a lab machine
        data_root = '/mnt/data/datasets/bugs_annotated_new'
        pass
    else:
        # assume we're on my personal machine
        data_root = '/data1/bugs_annotated_new'

    train_path_pos = os.path.join(data_root,
                                  'Good Images/Positive Counts/Training Set/')
    train_path_neg = os.path.join(data_root,
                                  'Good Images/No Counts/Training_Set')
    test_path_pos = os.path.join(data_root,
                                 'Good Images/Positive Counts/Test Set/')
    test_path_neg = os.path.join(data_root,
                                 'Good Images/No Counts/Test_Set')

    return train_path_pos, train_path_neg, test_path_pos, test_path_neg
# end def get_paths



def rot_img_array(input_array, kind=1):

    # work with input array with any dimension
    # given that dim -2 is rows and dim -1 is cols

    if kind == 0 or kind == 'original':
        return input_array

    elif kind == 1 or kind == 'rot90':
        return input_array[..., ::-1].swapaxes(-1, -2)
        
    elif kind == 2 or kind == 'rot180':
        return input_array[..., ::-1, ::-1]
    
    elif kind == 3 or kind == 'rot270':
        return input_array.swapaxes(-1, -2)[..., ::-1]
        
    elif kind == 4 or kind == 'fliplr':
        return input_array[..., ::-1]
    
    elif kind == 5 or kind == 'transpose':
        return input_array.swapaxes(-1, -2)
            
    elif kind == 6 or kind == 'flipud':
        return input_array[..., ::-1, :]

    elif kind == 7 or kind == 'altertrans':
        return input_array[..., ::-1, ::-1].swapaxes(-1, -2)

    else:
        raise ValueError('''kind can only be 0~7 or one of [original, fliplr, flipud, rot180, transpose, rot90, rot270, altertrans]''')


def x1y1x2y2_to_x1y1wh_single(bb):
    width = bb[2] - bb[0]
    height = bb[3] - bb[1]
    xy = (bb[0], bb[1])
    return (xy, width, height)


def x1y1x2y2_to_x1y1wh_batch(bbs):
    return [x1y1x2y2_to_x1y1wh_single(bb) for bb in bbs]


def x1y1wh_to_x1y1x2y2_single(bb):
    (x, y), width, height = bb
    return (x, y, x + width, y + height)


def x1y1wh_to_x1y1x2y2_batch(bbs):
    return [x1y1wh_to_x1y1x2y2_single(bb) for bb in bbs]


def function():
    pass



if __name__ == '__main__':
    pass
