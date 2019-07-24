'''
For releasing standalone detection script
'''


import os
import sys
import shutil

dst_dir = sys.argv[1]
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)

copy_list = [
    'detection.py',
    'tools.py',
    '__init__.py',
    'convnet_class.py',
    'tools_theano.py',
    'fileop.py',
    'layers.py',
    'fit.py',
    'detect.py',
]

for src_file in copy_list:
    dst_file = os.path.join(dst_dir, src_file)
    shutil.copy(src_file, dst_file)
