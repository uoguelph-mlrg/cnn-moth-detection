""" Note: currently does not catch moths that are overlapping with grids
They are joined with grid contours and become too big """

import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
# from tools_preproc import lcn_2d
import os

imargs = {'cmap': 'gray', 'interpolation': 'none'}

# gray_p = lcn_2d(gray)

# # adaptiveThreshold wants an 8-bit grayscale image
# alph = 255.0/(gray_p.max() - gray_p.min())
# bet =  -gray_p.min() * 255.0/(gray_p.max() - gray_p.min())
# g_s = np.cast[np.uint8](gray_p * alph + bet)

# plt.imshow(gray, cmap='gray', interpolation='nearest')


def get_contours(img, min_contour_area=5, max_contour_area=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray_sm = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(gray_sm, 255, 1, 1, 11, 2)


    new_thresh = thresh.copy()  # findContours overwrites input

    contours, hierarchy = cv2.findContours(new_thresh, cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    if 0:
        fig, subs = plt.subplots(nrows=2, ncols=2, num=1)


        for s in subs.flatten():
            s.get_xaxis().set_visible(False)
            s.get_yaxis().set_visible(False)

        subs[0, 0].imshow(gray, **imargs)
        # subs[0,1].imshow(gray_p, **imargs)
        subs[0, 1].imshow(gray_sm, **imargs)
        subs[1, 0].imshow(thresh, **imargs)

        plt.tight_layout()



    # get area of contours
    c_area = np.asarray([cv2.contourArea(c) for c in contours])

    keep_idx = (c_area >= min_contour_area) & (c_area <= max_contour_area)

    c_area = c_area[keep_idx]
    contours = np.asarray(contours)[keep_idx]

    return contours, c_area
# end def get_contours


if __name__ == "__main__":
    train_path = '/mnt/data/datasets/bugs_annotated_2014/new_separation/train/combined/'

    jpg_train = [f for f in os.listdir(train_path) if f.find('.jpg') > 0]

    for i, j in enumerate(jpg_train[:10]):
        try:
            img = scipy.misc.imread(os.path.join(train_path, j))
            #img =  cv2.imread(j)
        except IOError:
            print "There was a problem reading the jpg: %s." % j
            continue


        contours, c_area = get_contours(img)

        c_idx = np.argsort(c_area)[::-1]  # biggest first



        # let's plot some contours
        fig1, subs1 = plt.subplots(nrows=1, ncols=1)

        cv2.drawContours(img, np.asarray(contours)[c_idx[:20]], -1, (0, 255, 0), -1)
        #cv2.drawContours(img, contours[c_idx[22]], -1, (0, 0, 255), 3)
        #cv2.drawContours(img, contours[c_idx[23]], -1, (255, 0, 0), 3)

        subs1.imshow(img, **imargs)
        subs1.set_title(j)

        plt.tight_layout()


    # img_path = '/data1/bugs_annotated_new/Good Images/Positive Counts/Training Set/120_5096.jpg'
