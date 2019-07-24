'''
For copying random test images to the figure folder
'''
import os
import random
import shutil

list_src_dir = [
    '/mnt/data/wding/tmp/bugs/single_21/plots_round-1_test_0.823',
    '/mnt/data/wding/tmp/bugs/single_35/plots_round-1_test_0.998']

dst_dir = raw_dir = os.path.expanduser('~/Dropbox/automoth_paper_figures/')
list_sub_dir = ['more_examples_thresh_obj', 'more_examples_thresh_img']
# list_ex_typical = ['det_example_obj.pdf', 'det_example_img.pdf']
# img_typical = '1239_36080.jpg.pdf'
list_ex_typical = ['det_example_obj.jpg', 'det_example_img.jpg']
img_typical = '1239_36080.jpg'

for src_dir, sub_dir, ex_typical in \
        zip(list_src_dir, list_sub_dir, list_ex_typical):
    shutil.copyfile(os.path.join(src_dir, img_typical),
                    os.path.join(dst_dir, ex_typical))

    num_copy = 14

    img_list = sorted([item for item in os.listdir(src_dir)
                       if '.jpg.pdf' in item or '.jpg' in item])
    img_list.remove(img_typical)
    random.seed(0)
    indices_rand = sorted(random.sample(range(len(img_list)), num_copy))

    for ind in range(len(indices_rand)):
        src_name = os.path.join(src_dir, img_list[indices_rand[ind]])
        dst_file_name = ('ex_{:02}.pdf' if src_name.endswith('pdf')
                         else 'ex_{:02}.jpg')
        dst_name = os.path.join(dst_dir, sub_dir,
                                dst_file_name.format(ind + 1))
        shutil.copyfile(src_name, dst_name)
