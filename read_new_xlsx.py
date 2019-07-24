'''
Making labels from the xlsx files from the 2014/12 new data.

Information to extract

sequence information

labels in each image


each column

5: image name
6: coordinate label


'''
import xlrd
import os
import scipy.misc
import matplotlib.pylab as plt
import numpy as np
import cPickle as pkl


def get_info(xlsx_dir):
    '''
    TODO: extract more useful information
    '''
    flag_onlyfirst = True

    xlsx_list = [file_name for file_name in os.listdir(xlsx_dir)
                 if '.xlsx' == file_name[-5:]]

    label_dict = {}

    for xlsx_name in xlsx_list:
        xlsx_path = os.path.join(xlsx_dir, xlsx_name)
        workbook = xlrd.open_workbook(xlsx_path)
        worksheet = workbook.sheet_by_index(0)
        num_rows = worksheet.nrows

        for ind in range(1, num_rows):
            base_status = 0

            if flag_onlyfirst:
                base_status = int(worksheet.cell_value(ind, 7))

            if not base_status:

                img_name = worksheet.cell_value(ind, 5)
                label_str = worksheet.cell_value(ind, 6)
                xy_list = [[int(val) for val in item.split(',')]
                           for item in label_str.split('|')] \
                    if len(label_str) else []

                label_dict[img_name] = xy_list

    return label_dict


def show_image(img_dir, img_name, label_dict):

    img_path = os.path.join(img_dir, img_name)

    img = scipy.misc.imread(img_path)

    plt.figure()
    plt.imshow(img)
    if len(label_dict[img_name]):
        points = np.array(label_dict[img_name]) / 100. * [640., 480.]
        plt.hold('on')
        plt.scatter(points[:, 0], points[:, 1])
    plt.show()


if __name__ == '__main__':

    xlsx_dir = "/mnt/data/datasets/automoth_1412"

    label_dict = get_info(xlsx_dir)

    num_moth_dict = {}
    num_moth_list = []
    img_name_list = sorted(label_dict.keys())

    for img_name in img_name_list:
        num_moth_dict[img_name] = len(label_dict[img_name])
        num_moth_list.append(len(label_dict[img_name]))

    img_name = img_name_list[np.argmax(num_moth_list)]


    # label_path = os.path.join(xlsx_dir, 'labels.pkl')
    # with open(label_path, 'wb') as f:
    #     pkl.dump(label_dict, f)

    xlsx_path = "/mnt/data/datasets/automoth_1412/Some_Farms.xlsx"
    img_dir = "/mnt/data/datasets/automoth_1412/imgs"
    # img_name = '6272_186176.jpg'
    # img_name = '2985_126898.jpg'

    show_image(img_name=img_name, img_dir=img_dir, label_dict=label_dict)

    img_ind = np.where(np.array(img_name_list) == img_name)[0][0]

    show_image(img_name=img_name_list[img_ind - 1], img_dir=img_dir, label_dict=label_dict)
    show_image(img_name=img_name_list[img_ind - 4], img_dir=img_dir, label_dict=label_dict)


    # if only use the first image of each sequence (578 sequences in total)
    # then only have 27 images with moth and 155 moths
