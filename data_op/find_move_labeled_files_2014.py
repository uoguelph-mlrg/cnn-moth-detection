'''
This script find all the label files and their corresponding images and copy 
them to the same directory.
Specifically, it copies the original data to bugs_annotated_2014/Good Images/All


Another script (separate_no_pos_train_test_2014.py) following this will separate these images and label files into
no counts/postive counts and training/test.


'''

import os
import shutil

DEBUG = False

# if __name__ == "__main__":

dir_src_root = '/export/mlrg/wding/Data/bugs_2014'
dir_tar_root = '/mnt/data/datasets/bugs_annotated_2014/Good Images'
if not os.path.isdir(dir_tar_root):
    os.makedirs(dir_tar_root)

dir_tar_all = os.path.join(dir_tar_root, 'All')
if not os.path.isdir(dir_tar_all):
    os.makedirs(dir_tar_all)


# annotation files
dir_src_ann = os.path.join(dir_src_root, 'Annotation Files')
dir_src_img = os.path.join(dir_src_root, 'Pictures')


dirlist_ann = os.listdir(dir_src_ann)
dirlist_img = os.listdir(dir_src_img)


# for every annotation dir, find the corresponding image dir, find the image 
# corresponding to the annotation file, move the to the same folder, 

num_ann = 0

for folder_ann in dirlist_ann:
    filelist_ann = os.listdir(os.path.join(dir_src_ann, folder_ann))
    
    # record how many 
    num_ann = num_ann + len(filelist_ann)
    
#    if DEBUG:
#        print folder_ann.split()[-1].lower()
    
    # find the matching image folder
    for i, folder_img in enumerate(dirlist_img):
        

        # if matched, need to go into the folder and match files
        
        
        if DEBUG:
            print i
            print dirlist_img
            print folder_img
            print folder_img.split('-')[-1].lower().split()[-1]
            print folder_ann.split()[-1].lower()
        
        
        if folder_img.split('-')[-1].lower().split()[-1] == folder_ann.split()[-1].lower():
            
            if DEBUG:
                print len(os.listdir(os.path.join(dir_src_ann, folder_ann)))
                print len(os.listdir(os.path.join(dir_src_img, folder_img, 'Good Pictures')))
            
            filelist_img = os.listdir(os.path.join(dir_src_img, folder_img, 'Good Pictures'))
            
            for file_ann in filelist_ann:
                
                # need to match the image file name, if matched copy them out 
                # together
                for file_img in filelist_img:
                
                    if file_img[:-4] == file_ann:
                        shutil.copyfile(os.path.join(dir_src_ann, folder_ann, file_ann), 
                                        os.path.join(dir_tar_all, file_ann)
                                        )
                                        
                        shutil.copyfile(os.path.join(dir_src_img, folder_img, 'Good Pictures', file_img), 
                                        os.path.join(dir_tar_all, file_img)
                                        )
                                        
                        
                        break  # for file_img
                    
            break  # for folder_img
