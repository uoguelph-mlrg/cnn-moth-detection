'''
Show labeled moths and image statistics
'''

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import colorsys
import yaml

from annotation import get_annotation, get_bbs
from tools_plot import dispims

plt.ion()

with open('config.yaml', 'r') as f:
    config = yaml.load(f)

train_path = os.path.join(config['data_path'], config['train_subpath_pos'])
test_path = os.path.join(config['data_path'], config['test_subpath_pos'])

jpg_train = [f for f in os.listdir(train_path) if f.find('.jpg') > 0]
jpg_test = [f for f in os.listdir(test_path) if f.find('.jpg') > 0]

# imargs = {'cmap': 'gray',}
imargs = {'interpolation': 'none'}
plt.close('all')

moth_list = []
for i, j in enumerate(jpg_train + jpg_test):
    try:
        im = scipy.misc.imread(os.path.join(train_path, j))
    except IOError:
        print "There was a problem reading the jpg: %s." % j
        continue

    # the rollaxis command rolls the last (-1) axis back until the start
    # do a colourspace conversion
    im_y, im_i, im_q = colorsys.rgb_to_yiq(*np.rollaxis(im[..., :3], axis=-1))
    ann_file = j.split('.')[0]
    ann_path = ann_path = os.path.join(train_path, ann_file)
    annotation = get_annotation(ann_path)

    # get all bbs for this image
    bbs = get_bbs(annotation)
    for xy, width, height in bbs:
        x, y = xy
        # remember y is indexed first in image
        moth = im_y[y:(y + height), x:(x + width)]
        moth_list.append(moth)

n_moths = len(moth_list)
widths = np.array([m.shape[1] for m in moth_list])
max_w = widths.max()
heights = np.array([m.shape[0] for m in moth_list])
max_h = heights.max()

moth_array = np.zeros((n_moths, max_h, max_w))

for i, m in enumerate(moth_list):
    moth_array[i, :m.shape[0], :m.shape[1]] = m

dispims(moth_array.reshape(n_moths, max_h * max_w).T, max_h, max_w, border=2)

# add a maxdim to represent the max dimension between height and widths
maxdim = heights * (heights >= width) + width * (heights < width)


# fig.patch.set_alpha(0)
# ax.patch.set_alpha(0)

# # transparent figure with totally tight axes
# # http://stackoverflow.com/a/3133953
# fig=figure()
# ax=fig.add_axes((0,0,1,1))
# ax.set_axis_off()
# ax.plot([3,1,1,2,1])
# ax.plot([1,3,1,2,3])
# plt.savefig('/tmp/test.pdf', transparent=True)

# # transparent figure with totally tight axes
# fig=figure()
# ax=fig.add_axes((0,0,1,1))
# ax.set_axis_off()
# ax.imshow(im)
# plt.savefig('/tmp/test.pdf', transparent=True)

# transparent figure with totally tight axes
# the top and left edges seem to get cropped a bit
#fig=figure()
#ax=fig.add_axes((0,0,1,1))
#ax.set_axis_off()
#dispims(moth_array.reshape(n_moths, max_h * max_w).T, max_h, max_w, border=2),
#plt.axis('tight')
#plt.savefig('/tmp/test.pdf', transparent=True)

# logarithmic x-scale histogram with customized tickers
# http://matplotlib.org/api/ticker_api.html#matplotlib.ticker.ScalarFormatter
# http://stackoverflow.com/a/6856155
fig_h = plt.figure()
s = fig_h.add_subplot(141)
s.hist(widths, bins=np.logspace(1, 6, base=2))
s.set_xscale("log", basex=2)
s.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%6.f'))
s.set_title('widths')
s = fig_h.add_subplot(142)
s.hist(heights, bins=np.logspace(1, 6, base=2))
s.set_xscale("log", basex=2)
s.set_title('heights')
s.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%6.f'))
s = fig_h.add_subplot(143)
s.hist(1.0 * widths / heights, bins=np.logspace(-3, 3, base=2))
s.set_xscale("log", basex=2)
s.set_title('aspect ratio (w/h)')
s.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%6.1f'))
s = fig_h.add_subplot(144)
s.hist(np.maximum(heights, widths), bins=np.logspace(1, 6, base=2))
s.set_xscale("log", basex=2)
s.set_title('Max dimension')
s.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter('%6.f'))
