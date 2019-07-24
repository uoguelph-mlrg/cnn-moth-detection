'''
This script is to deal with newly labeled data by Jordan in march 2014.

It is adopted from find_move_labeled_files_2014

All the images copied here will be used as test images
'''

import os
import shutil

DEBUG = False

# if __name__ == "__main__":

dir_src_root = '/export/mlrg/wding/Data/bugs_2014'
dir_tar_root = '/export/mlrg/wding/Data/bugs_2014/New_pairs_Feb27'
# dir_tar_root = '/mnt/data/datasets/bugs_annotated_2014/Good Images'
if not os.path.isdir(dir_tar_root):
    os.makedirs(dir_tar_root)

dir_tar_all = os.path.join(dir_tar_root, 'All')
if not os.path.isdir(dir_tar_all):
    os.makedirs(dir_tar_all)


# annotation files
dir_src_ann = os.path.join(dir_src_root, 'New Annotations_ Feb 27')
dir_src_img = os.path.join(dir_src_root, 'Pictures')


dirlist_ann = os.listdir(dir_src_ann)
dirlist_img = os.listdir(dir_src_img)


# for every annotation dir, find the corresponding image dir, find the image 
# corresponding to the annotation file, move the to the same folder, 


# for each annotation file, look through all the files in picture folder

num_ann = 0

filelist_ann = os.listdir(dir_src_ann)

# record how many 
num_ann = num_ann + len(filelist_ann)


# iterate over different folders
for file_ann in filelist_ann:
    
    
    for folder_img in dirlist_img:    

        filelist_img = os.listdir(os.path.join(dir_src_img, folder_img, 'Good Pictures'))
        
        # need to match the image file name, if matched copy them out 
        # together
        for file_img in filelist_img:
        
            if file_img[:-4] == file_ann:
                shutil.copyfile(os.path.join(dir_src_ann, file_ann), 
                                os.path.join(dir_tar_all, file_ann)
                                )
                                
                shutil.copyfile(os.path.join(dir_src_img, folder_img, 'Good Pictures', file_img), 
                                os.path.join(dir_tar_all, file_img)
                                )
