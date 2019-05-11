import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
from keras.models import model_from_json
from keras.models import load_model
import matplotlib.pyplot as plt
from PIL import Image


#Loading the Training and Test data
path_s='D:\\Grain'
train_data=path_s+'\\train_data'
dataset=path_s+'\\dataset'
test_data=path_s+'\\test images'

#Image Preprocessing
def Data_Pre_Processing(mypath,data):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    nm=1
    id_final=str(data.split('.')[0])
    for files in tqdm(onlyfiles):
        if nm < 550:
            file=mypath+"\\"+files
            
            im = Image.open(file)
            im.thumbnail((1024,1024), Image.ANTIALIAS)
            new_file=path_s+"//image.jpg"
            im.save(new_file,"JPEG")
            grey=cv2.imread(new_file,cv2.IMREAD_GRAYSCALE)
            credential=id_final+"."+str(nm)
            image_path=dataset+'\\'+credential+".jpg"
            resized_image = cv2.resize(grey, (64,64)) 
            cv2.imwrite(image_path,resized_image)
            nm=nm+1
            
            
def main_function(train_data):
    os.chdir(train_data)
    all_subdirs = [d for d in os.listdir('.') if os.path.isdir(d)]
    for dirs in tqdm(all_subdirs):
        dir = os.path.join(train_data,dirs)
        os.chdir(dir)
        path = os.getcwd()
        tags = str(path).split("\\")[3]
        Data_Pre_Processing(path,tags)
main_function(train_data)         
#Assigning Label 
def one_hot_lable(img):
    #ohl=[0,0]
    lable=img.split('.')[0]
    if lable == 'positive':
        ohl=np.array(1)#positive
    else:
        ohl=np.array(2)#negative
    return ohl

#Assingin label for Training data
def train_data_with_lable():
    train_image=[]
    for i in tqdm(os.listdir(dataset)):
        path=os.path.join(dataset,str(i))
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(64,64))
        train_image.append([np.array(img),one_hot_lable(i)])
    shuffle(train_image)
    return train_image

#Assigning label for Test data
def test_data_with_lable():
    test_image=[]
    for i in tqdm(os.listdir(test_data)):
        path=os.path.join(test_data,str(i))
        img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
        img=cv2.resize(img,(64,64))
        test_image.append([np.array(img),one_hot_lable(i)])
    return test_image
   
training_image = train_data_with_lable()
testing_image = test_data_with_lable()


tr_img_data = np.array([i[0] for i in training_image]).reshape(-1,64,64,1)
tr_lbl_data = np.array([i[1] for i in training_image])
tst_img_data = np.array([i[0] for i in testing_image]).reshape(-1,64,64,1)
tst_lbl_data = np.array([i[1] for i in testing_image])


#Implementing CNN Model

model=Sequential()

#Applying first Convolutional layer
model.add(Conv2D(input_shape=[64,64,1],filters=32,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

#Applying Second Convolutional Layer
model.add(Conv2D(filters=45,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))

#Applying Third Convolutional Layer
model.add(Conv2D(filters=100,kernel_size=5,strides=1,padding='same',activation='relu'))
model.add(MaxPool2D(pool_size=5,padding='same'))
model.add(Dropout(0.25))
#Applying Flatten
model.add(Flatten())
model.add(Dense(15,activation='relu'))
model.add(Dropout(rate=0.05))
model.add(Dense(3,activation='softmax'))
optimizer=Adam(lr=1e-3)

model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x=tr_img_data,y=tr_lbl_data,epochs=5,batch_size=200)
model.summary()
loss, acc = model.evaluate(tst_img_data,tst_lbl_data,verbose=0,steps=10)
print('loss=',loss)
print('accuracy=',acc*100)

#
fig=plt.figure(figsize=(14,14))

#Function for Total Grain Count
G_count=0
NTG_count=0
for cnt,data in enumerate(testing_image[0:len(testing_image)]):
    y=fig.add_subplot(6,5,cnt+1)
    img=data[0]
    data=img.reshape(1,64,64,1)
    model_out=model.predict([data])
    if np.argmax(model_out) == 1:
        str_lable='Grain'
        G_count+=1
    else:
        str_lable='Not Grain'
        NTG_count+=1
    y.imshow(img,cmap='gray')
    plt.title(str_lable)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
    plt.savefig('D:/Grain')

print("grain count=",G_count)
print("Not grain count=",NTG_count)
