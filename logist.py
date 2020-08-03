import numpy as np
from PIL import Image
import os
from os.path import splitext
from os import listdir
import cv2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#from sklearn.model_selection import tr
def randomfile():
    num =np.arange(1200)
    np.random.shuffle(num)
    num1 =num[:1000]
    num2 =num[1000:]

    filename1 ='train.txt'
    filename2 ='trainlabel.txt'
    filename3 ='val.txt'
    filename4 ='vallabel.txt'
    dir_img = 'J:/game/seg_classification/data/'
    dir_mask ='J:/game/seg_classification/label/'
    image =listdir(dir_img)
    #label -listdir(dir_mask)
    np.random.shuffle(image)
    image1 =image[:1000]
    image2 =image[1000:]
    with open(filename1,'w') as f1:
        for i in range(len(image1)):
            f1.write(os.path.join("J:/game/seg_classification/data/",str(image1[i])+'\n'))
    with open(filename2,'w') as f2:
        for i in range(len(image1)):
            label=splitext(image1[i])[0]
            f2.write(os.path.join('J:/game/seg_classification/label/',str(label)+'_mask.png\n'))
    with open(filename3,'w') as f3:
        for i in range(len(image2)):
            f3.write(os.path.join("J:/game/seg_classification/data/",str(image1[i])+'\n'))
    with open(filename4,'w') as f4:
        for i in range(len(image2)):
            label=splitext(image2[i])[0]
            f4.write(os.path.join('J:/game/seg_classification/label/',str(label)+'_mask.png\n'))
    print(image)

#randomfile()
def read_data():
    train_data=[]
    labels=[]
    fo1 =open("train.txt",'r')
    fo2 =open("trainlabel.txt",'r')
    for line in fo1.readlines():
        line = line.rstrip()
        img =Image.open(line)

        img=img.resize((200,200))
        img=np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).ravel()
        #print(img.shape)
        train_data.append(img)
    fo1.close()
    for line in fo2.readlines():
        line =line.rstrip()
        label =Image.open(line)
        label =label.resize((200,200))
        label =np.array(label).ravel()
        for i in range(len(label)):
            if label[i]==128:
                label[i]=1
            if label[i]==255:
                label[i]=2
        labels.append(label)
    fo2.close()

    return train_data, labels

def read_val():
    val_data=[]
    labels=[]
    fo1 =open("val.txt",'r')
    fo2 =open("vallabel.txt",'r')
    for line in fo1.readlines():
        line = line.rstrip()
        img =Image.open(line)
        img=img.resize((200,200))
        img=np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).ravel()
        val_data.append(img)
    fo1.close()
    for line in fo2.readlines():
        line =line.rstrip()
        label =Image.open(line)
        label =label.resize((200,200))
        label =np.array(label).ravel()
        labels.append(label)
    fo2.close()
    return val_data, labels

if __name__=='__main__':
    X_train,Y_train=read_data()

    X_val,Y_val =read_val()
    #print(np.array(X_train[1]).shape)
    #print(Y_train[1])
    #Regression
    #X_train=np.array(X_train)
    #msample,nx,ny =X_train.shape
    #X_train= X_train.reshape((msample,nx*ny))
    #print(X_train.shape)
    #X_train.tolist()
    #Y_train =np.array(Y_train)

    #nsample,nx1,ny1 =Y_train.shape
    #Y_train = Y_train.reshape((nsample, nx1 * ny1))
    #print(Y_train.shape)
    #Y_train.tolist()
    logreg =LogisticRegression()
    logreg.fit(X_train,Y_train)
    
    print(logreg)
    #训练集上准确率
    print("accuracy:",accuracy_score(logreg.predict(X_train),Y_train)) 
