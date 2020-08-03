import os 
from shutil import copyfile
from os import listdir


save_dir =r'J:/game/seg_classification/data/'
imgs_dir =r'J:/game/seg_classification/_ouput_dir_/'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)


for files in listdir(imgs_dir):
    if files[-5] =='0':
       source_file=os.path.join(imgs_dir +files)
       target_file =os.path.join(save_dir +files)
       copyfile(source_file,target_file) 

 
