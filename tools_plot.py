'''
For plotting curves after run_pipeline
'''
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter
# previously tried LinearLocator, LogLocator
import numpy as np
import os
from fileop import loadfile
import logging

from collections import OrderedDict
import matplotlib
matplotlib.use('Agg')

logger = logging.getLogger(__name__)


def dispims(M, height, width, border=0, bordercolor=0.0, layout=None, **kwargs):
    """ Display a whole stack (colunmwise) of vectorized matrices. Useful
        eg. to display the weights of a neural network layer.
    """
    numimages = M.shape[1]
    if layout is None:
        n0 = int(np.ceil(np.sqrt(numimages)))
        n1 = int(np.ceil(np.sqrt(numimages)))
    else:
        n0, n1 = layout
    im = bordercolor * np.ones(((height + border) * n0 + border, (width + border) * n1 + border), dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i * n1 + j < M.shape[1]:
                im[i * (height + border) + border:(i + 1) * (height + border) + border,
                   j * (width + border) + border:(j + 1) * (width + border) + border] = \
                    np.vstack((np.hstack((np.reshape(M[:, i * n1 + j], (height, width)),
                              bordercolor * np.ones((height, border), dtype=float))),
                              bordercolor * np.ones((border, width + border), dtype=float)
                               )
                              )
                               
    plt.imshow(im, cmap=plt.cm.gray, interpolation='nearest', **kwargs)
    # plt.show()
# end def dispims


def annotate_bbs(ax, bbs, clrstr='blue', textlabels=None, label=None):
    """ Draw bounding boxes on axis
    bbs is an N x 4 array
    each row is x1, y1, x2, y2"""
    for qq, (x1, y1, x2, y2) in enumerate(bbs):
        # plt.axhspan(y_i, y_i + target_height, x_i, x_i + target_width,
        #             facecolor='red', alpha=0.5)
        w = x2 - x1
        h = y2 - y1
        # note for drawing commands, width is first, height is second
        ax.add_patch(mpl.patches.Rectangle((x1, y1), w,
                                           h, fill=False,
                                           ec=clrstr,
                                           label=label))
        if textlabels is not None:
            ax.text(x1 + w, y1, '%d' % textlabels[qq], fontsize='x-small',
                    ha='right', va='top')
# end def annotate_bbs


def plot_fppi_mr_curves(ax, fppi_mr_list,
                        color_list=['b', 'r', 'y', 'm', 'k', 'c', 'g'],
                        linestyle_list=['--', '-.', '-'],
                        marker_list=['+', '.', '*', 'd'],
                        markersize=7,
                        flag_marker=False):

    """
    For plotting multiple fppi_mr curves in same plot.
    Input
    fig, figure handle
    fppi_mr_list, is a list. Each fppi_mr = fppi_mr_list[i] is a 2D numpy array.
    fppi_mr[j, 0] and fppi_mr[j, 1] make a pair of false positive per image and
    missing rate.

    properties of figure
    color_list
    linestyle_list
    marker_list
    markersize
    """

    plt.hold('on')
    for ind_retrain, fppi_mr in enumerate(fppi_mr_list):

        # ax.loglog(fppi_mr[:, 0], fppi_mr[:, 1])
        
        if flag_marker:
            marker = marker_list[ind_retrain % len(marker_list)]
        else:
            marker = None

        ax.semilogx(fppi_mr[:, 0], fppi_mr[:, 1],
                    color=color_list[ind_retrain % len(color_list)],
                    label='round' + str(ind_retrain),
                    marker=marker,
                    linestyle=linestyle_list[ind_retrain % len(linestyle_list)],
                    markersize=markersize)

    majorFormatter = FormatStrFormatter('%.2f')
    # The following line need to be modified as the y axis is not log scale
    majorLocator = FixedLocator(np.linspace(0, 1, 11))

    ax.yaxis.set_major_locator(majorLocator)
    ax.yaxis.set_major_formatter(majorFormatter)
    ax.set_ylim(0, 1)
    ax.set_xlim(0.01, 500)
    ax.set_xlabel('false positives per image')
    ax.set_ylabel('miss rate')


    plt.legend(loc='lower left')
# end def plot_fppi_mr_curves


def plot_fppi_mr_traj(ax, fppi_list, mr_list):
    # ax.plot(fppi_list, mr_list)

    for ind in range(len(fppi_list) - 1):
        # ax.arrow(fppi_list[ind], mr_list[ind],
        #          fppi_list[ind + 1] - fppi_list[ind],
        #          mr_list[ind + 1] - mr_list[ind])
        ax.annotate("",
                    xy=(fppi_list[ind + 1], mr_list[ind + 1]), xycoords='data',
                    xytext=(fppi_list[ind], mr_list[ind]), textcoords='data',
                    arrowprops=dict(arrowstyle="->",  # linestyle="dashed",
                                    color="0.5",
                                    shrinkA=0, shrinkB=0,
                                    patchA=None,
                                    patchB=None,
                                    ),
                    )
# end def plot_fppi_mr_traj






def generate_annotations(fppi_orig, mr_orig, config,
                         fppi_range=[0., 10.],
                         ann_fppi_list=[1, 5, 10]):

    # calculate the interpolation function
    perm_ind = np.argsort(fppi_orig)
    fppi_sorted = fppi_orig[perm_ind]
    mr_sorted = mr_orig[perm_ind]

    # This scipy version cannot do extrapolating conveniently
    # fcn_interp = interp1d(np.log(fppi_sorted), mr_sorted)

    # extrapolating with the end point value
    fcn_interp = lambda input_list: np.interp(input_list,
                                              np.log(fppi_sorted),
                                              mr_sorted)

    # calcualte log average MR, will move to another place
    logspace_fppi = 10 ** np.linspace(fppi_range[0], np.log10(fppi_range[1]), 11)
    log_avg_mr = np.mean(fcn_interp(np.log(logspace_fppi)))

    # fppi and mr to be annotated
    ann_mr_list = fcn_interp(np.log(ann_fppi_list))


    # generate annotate texts

    text_tofill = '''
    Log average mr:\n    %.2f between fppi %d and %d\n
    Training set:\n    %s
    Validation set:\n    %s
    Test set:\n    %s
    If multiple scale:\n    %s
    Scales: \n    %s
    '''

    params_tuple = (log_avg_mr,
                    fppi_range[0],
                    fppi_range[1],
                    config['detect_train_set'],
                    config['detect_valid_set'],
                    config['detect_test_set'],
                    config['flag_multiscale'],
                    config['detect_width_list'] if config['flag_multiscale'] else config['target_width'])

    ann_text = text_tofill % params_tuple

    return ann_mr_list, ann_text
# end def generate_annotations



def add_annotations(ax,
                    ann_fppi_list=[1, 5, 10],
                    ann_mr_list=[0.8, 0.6, 0.4],
                    len_arrow=4.,
                    ann_text_left=60,
                    ann_text_bottom=0.5,
                    ann_text=''):
    '''
    todo:
        write comments

    ann_fppi_list
    ann_mr_list
    len_arrow  # note that this is log scale length
    ann_text_left
    ann_text_bottom
    ann_text
    '''

    # annotate the list of fppi and mr
    for ind in range(len(ann_fppi_list)):
        plt.plot([ann_fppi_list[ind], ann_fppi_list[ind]],
                 [0, ann_mr_list[ind]], linestyle='--', color='k')

        ax.annotate("mr=%.2f\nfppi=%d" % (ann_mr_list[ind],
                                          ann_fppi_list[ind]),
                    xy=(ann_fppi_list[ind], ann_mr_list[ind]),
                    xycoords='data',
                    xytext=(ann_fppi_list[ind] / len_arrow, ann_mr_list[ind]),
                    textcoords='data',
                    arrowprops=dict(arrowstyle="->",  # linestyle="dashed",
                                    color="0.5",
                                    shrinkA=0, shrinkB=0,
                                    patchA=None,
                                    patchB=None,
                                    )
                    )

    # add the side annotation text
    ax.text(ann_text_left, ann_text_bottom, ann_text)
# end def add_annotations


def generate_result_figure(run_path='', data_set='train', dict_record=None, config=None, flag_debug=False, flag_traj=False, ext='.png'):

    '''
    run_path should be given, except for the case that:
        both dict_record and config are given and flag_debug is True

    data_set should be train, valid or test
    '''


    if dict_record is None:
        # dict_record is a dictionary containing the results
        dict_record = loadfile(os.path.join(run_path, 'results.pkl'))

    if config is None:
        # config is a dictionary containing the parameters
        config = loadfile(os.path.join(run_path, 'config.yaml'))


    # common parameters for both test and train
    fppi_range = [0., 10.]
    ann_fppi_list = [1, 2, 4, 8, 16]
    ann_text_left = 60
    ann_text_bottom = 0.5
    len_arrow = 4.
    figsize = (20, 10)
    
    if data_set == 'train':
        fppi_mr_list = dict_record['fppi_mr_train_list']
    elif data_set == 'valid':
        fppi_mr_list = dict_record['fppi_mr_valid_list']
    elif data_set == 'test':
        fppi_mr_list = dict_record['fppi_mr_test_list']
    else:
        raise NotImplementedError(
            'data_set should only be train, valid or test')

    fppi_orig = fppi_mr_list[-1][:, 0]
    mr_orig = fppi_mr_list[-1][:, 1]
    ann_mr_list, ann_text = generate_annotations(fppi_orig,
                                                 mr_orig,
                                                 config,
                                                 fppi_range=fppi_range,
                                                 ann_fppi_list=ann_fppi_list)

    plt.close(1)
    fig = plt.figure(num=1, figsize=figsize)
    ax = fig.gca()

    plt.title('fppi-mr curve, {} set'.format(data_set))

    # fppi mr curves
    plot_fppi_mr_curves(ax, fppi_mr_list)

    if data_set == 'train':
        # training trajectories based on threshold selections
        if flag_traj:
            plot_fppi_mr_traj(ax, dict_record['fppi_chosen_list'],
                              dict_record['mr_chosen_list'])

    # annotate parameters
    add_annotations(ax,
                    ann_fppi_list=ann_fppi_list,
                    ann_mr_list=ann_mr_list,
                    len_arrow=len_arrow,
                    ann_text_left=ann_text_left,
                    ann_text_bottom=ann_text_bottom,
                    ann_text=ann_text)

    if flag_debug:
        plt.show()
    else:
        fig.savefig(os.path.join(
            run_path, 'fppi_mr_{}{}'.format(data_set, ext)))


# end def generate_result_figure





def imshow_wrap(sub, im):
    '''
    for wrapping common functions for plotting left and right
    '''
    imargs = {'cmap': 'gray', 'interpolation': 'none'}
    sub.cla()
    sub.imshow(im, **imargs)
    # sub.axis('off')
    plt.setp(sub.get_xticklabels(), visible=False)
    plt.setp(sub.get_yticklabels(), visible=False)


def make_legend_wrap(sub):
    # nice way to get rid of repeated labels in legend
    # http://stackoverflow.com/a/13589144
    handles, labels = sub.get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    sub.legend(by_label.values(), by_label.keys(),
               fontsize='xx-small', loc='lower left',
               fancybox=True, framealpha=0.3)


def concatenate_images(x_in, pad_size=1, num_row=10, num_col=10):
    x_pad = np.pad(x_in,
                   ((0, 0), (pad_size, pad_size),
                    (pad_size, pad_size), (0, 0)),
                   mode='constant', constant_values=0)

    x_cat = np.concatenate(
        [np.concatenate(x_pad[ind * num_row:(ind + 1) * num_row], axis=0)
         for ind in range(num_col)],
        axis=1)

    return x_cat





if __name__ == '__main__':

    run_path = '/mnt/data/wding/tmp/bugs/bug_run_2015-03-31_23-48-15_54'
    generate_result_figure(run_path, flag_debug=False, data_set='train')
    generate_result_figure(run_path, flag_debug=False, data_set='valid')
    generate_result_figure(run_path, flag_debug=False, data_set='test')
