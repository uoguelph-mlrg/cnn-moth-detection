# from contours import get_contours
import os
import matplotlib.pyplot as plt
from scipy.misc import imread
from annotation import get_annotation, get_bbs
from tools_plot import annotate_bbs


# def 



def xywh_to_x1y1x2y2(xywh):
    (x, y), w, h = xywh

    # return (x - w // 2, y - h // 2, x + w // 2, y + h // 2)
    return (x, y, x + w, y + h)


fig_dir = os.path.expanduser('~/Dropbox/automoth_paper/figs')

pos_dir = '/mnt/data/datasets/bugs_annotated_2014/Good Images/Positive_Counts_All'
neg_dir = '/mnt/data/datasets/bugs_annotated_2014/Good Images/No_Counts_All'

# pos_name = '1471_39381.jpg'
# pos_name = '1088_26202.jpg'
# pos_name = '1227_35266.jpg'

pos_name = '1227_39869.jpg'
neg_name = '501_30706.jpg'


pos_path = os.path.join(pos_dir, pos_name)
neg_path = os.path.join(neg_dir, neg_name)

img_pos = imread(pos_path)
img_neg = imread(neg_path)
bbs_pos = get_bbs(get_annotation(pos_path.strip('.jpg')))


plt.ioff()

fig = plt.figure()
ax = plt.subplot(1, 1, 1)
annotate_bbs(ax, map(xywh_to_x1y1x2y2, bbs_pos))
plt.imshow(img_pos)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "example_pos.png"))

# TODO: make the boundaries tighter
# plot_margin = 0.25
# x0, x1, y0, y1 = plt.axis()
# plt.axis((x0 - plot_margin,
#           x1 + plot_margin,
#           y0 - plot_margin,
#           y1 + plot_margin))

# plt.tight_layout(pad=0., h_pad=0., w_pad=0.)

# plt.savefig(os.path.join(fig_dir, "example_pos.png"),
#             bbox_inches='tight')

plt.figure()
plt.imshow(img_neg)
plt.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "example_neg.png"))


# plt.show()
