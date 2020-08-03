import numpy as np
from PIL import Image
import os
from os.path import splitext
from os import listdir
import cv2
filename='bili3.txt'
fo1 = open("train.txt", 'r')
fo2 = open("trainlabel.txt", 'r')
dir_mask = 'J:/game/seg_classification/label/'
count=0
import csv
csvfile= open('test2.csv','w',encoding='utf-8',newline='')

writer=csv.writer(csvfile)
writer.writerow(['label','data'] )
with open(filename, 'w') as f1:
    for labelname in listdir(dir_mask):
    #for line in fo2.readlines():
        count=count+1
        count1 = 0
        count2 = 0
        count3 = 0
        #line = line.rstrip()
        label = Image.open('J:/game/seg_classification/label/'+labelname)
        label = np.array(label).ravel()
        print(len(label))
        for i in range(len(label)):
            if label[i]==0:
                count1=count1+1
            if label[i] == 128:
                count2=count2+1
            if label[i] == 255:
                count3=count3+1
        a1 =count1/count2
        a2 =count1/count3
        a3 =count2/count3
        print(count)
        writer.writerow([labelname[-10],a1])
        #f1.write(str(labelname[-10])+','+str(a1)+'\n')
    #labels.append(label)
#fo2.close()

