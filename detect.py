'''
Input: 
    1. folder path that contains images.
    2. folder path that outputs detection txt files.
    3. folder path that outputs illustrated images. If not specified, won't output images.

In each txt file, the first line is counts, followed by center locations


An optional 
see how to load detects
'''


import os
import sys
import time
import tempfile
import shutil

import yaml
import scipy.misc

from detection import detect


def bb_to_centers(bb):
    centers = []
    for item in bb:
        x = int((item[0] + item[2]) / 2.)
        y = int((item[1] + item[3]) / 2.)
        centers.append((x, y))
    return centers


def bb_to_lines(bb):
    '''
    Convert bounding boxes for one image to text lines.

    bb is a numpy ndarray. Each row represents one detection. 
    For each detection, 0 is x of top-left, 1 is y of top-left, 
    2 is x of bottom-right, 3 is y of bottom-right, 4 is the probability

    lines is a list. 0 is the count, i is the center coordinates of i-1'th detection.
    '''

    lines = []
    lines.append(str(len(bb)) + '\n')

    if len(bb) > 0:
        for item in bb:
            x = int((item[0] + item[2]) / 2.)
            y = int((item[1] + item[3]) / 2.)
            lines.append(str(x) + ' ' + str(y) + '\n')

    return lines


def label_centers(img, centers_list,
                  radius=10, color=(191, 0, 255), x_max=639, y_max=479):
    '''
    centers_list is a list containing tuples, each tuple is (x, y)
    '''
    for x, y in centers_list:
        img[max(y - 1, 0): min(y + 1, y_max),
            max(x - radius, 0): min(x + radius + 1, x_max), :] = color
        img[max(y - radius, 0): min(y + radius + 1, y_max),
            max(x - 1, 0): min(x + 1, x_max), :] = color
    return img


def label_bbs(img, bb_list,
              color=(191, 0, 255), x_max=639, y_max=479):
    pass


def detect_wrap(img_dir,
                clf_path=os.path.dirname(os.path.realpath(__file__)),
                config_path=os.path.dirname(
                    os.path.realpath(__file__)) + '/config.yaml',
                thresh=0.5):

    with open(config_path, 'r') as f:
        config = yaml.load(f)

    bbs = detect(data_path=img_dir,
                 write_path=clf_path,
                 target_width=config['target_width'],
                 target_height=config['target_height'],
                 x_stride=config['target_stride_x'],
                 y_stride=config['target_stride_y'],
                 thresh=thresh,
                 n_images=-1,
                 flag_rgb=config['flag_rgb'],
                 flag_usemask=False,
                 thresh_mask=config['thresh_mask'],
                 nms_thresh=config['nms_threshhold'],
                 flag_det_rot_aug=config['flag_det_rot_aug'],
                 )

    return bbs


def detect_img_wrap(img_path,
                    clf_path=os.path.dirname(os.path.realpath(__file__)),
                    config_path=os.path.dirname(
                        os.path.realpath(__file__)) + '/config.yaml',
                    thresh=0.5):
    '''
    doing detection on a single image. This is wrapper around the original detect_wrap function which does batch detection for a folder.
    '''

    img_name = os.path.basename(img_path)
    print 'detecting {img_name}...'.format(img_name=img_name)

    # create a temporary folder
    temp_dir = tempfile.mkdtemp(prefix='detector_')

    # copy the image into the temporary folder
    shutil.copyfile(img_path, os.path.join(temp_dir, img_name))

    # call detect_wrap to do detection
    bbs = detect_wrap(img_dir=temp_dir,
                      clf_path=clf_path,
                      config_path=config_path,
                      thresh=thresh)

    # delete temporary folder
    shutil.rmtree(temp_dir)


    return bbs




def main(img_dir, txt_dir=None, fig_dir=None,
         clf_path=os.path.dirname(os.path.realpath(__file__)),
         config_path=os.path.dirname(os.path.realpath(__file__)) + '/config.yaml', ):

    bgn_time = time.time()

    if not os.path.isdir(txt_dir):
        os.makedirs(txt_dir)

    bbs = detect_wrap(img_dir, config_path=config_path, clf_path=clf_path)

    for img_name in bbs.keys():
        lines = bb_to_lines(bbs[img_name])
        with open(os.path.join(txt_dir, img_name + '.txt'), 'w') as f:
            f.writelines(lines)

    if fig_dir:
        if not os.path.isdir(fig_dir):
            os.makedirs(fig_dir)

        for img_name in bbs.keys():
            img = scipy.misc.imread(os.path.join(img_dir, img_name))
            img = label_centers(img, bb_to_centers(bbs[img_name]))
            scipy.misc.imsave(os.path.join(fig_dir, 'det_' + img_name), img)

    return time.time() - bgn_time


if __name__ == '__main__':

    # img_dir = './original_images'
    # txt_dir = './txt_detects'
    # fig_dir = './labeled_images'

    if len(sys.argv) == 1:
        print \
            '''
python detect.py
    show help information
    
python detect.py img_dir txt_dir fig_dir
    detect moths from images stored in img_dir
    counts and moth locations will be stored in txt_dir
    images with moth labeled will be stored in fig_dir (if given)

python detect.py img_dir txt_dir
    won't save labeled images

python detect.py img_dir
    txt_dir will be the same as img_dir
        '''

    elif len(sys.argv) == 2:
        img_dir = sys.argv[1]
        txt_dir = img_dir
        fig_dir = None
    elif len(sys.argv) == 3:
        img_dir = sys.argv[1]
        txt_dir = sys.argv[2]
        fig_dir = None
    elif len(sys.argv) == 4:
        img_dir = sys.argv[1]
        txt_dir = sys.argv[2]
        fig_dir = sys.argv[3]

    if len(sys.argv) > 1:
        print main(img_dir=img_dir, txt_dir=txt_dir, fig_dir=fig_dir)
