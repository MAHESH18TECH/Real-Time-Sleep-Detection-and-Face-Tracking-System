#!/usr/bin/env python
# coding: utf-8

# # Creating the model

# In[ ]:


import os
from keras.preprocessing import image
import matplotlib.pyplot as plt 
import numpy as np
from keras.utils.np_utils import to_categorical
import random,shutil
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization
from keras.models import load_model


def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):

    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS= 32
TS=(24,24)
train_batch= generator('data/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch= generator('data/valid',shuffle=True, batch_size=BS,target_size=TS)
SPE= len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)


# img,labels= next(train_batch)
# print(img.shape)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
#32 convolution filters used each of size 3x3
#again
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),

#64 convolution filters used each of size 3x3
#choose the best features via pooling
    
#randomly turn neurons on and off to improve convergence
    Dropout(0.25),
#flatten since too many dimensions, we only want a classification output
    Flatten(),
#fully connected to get all relevant data
    Dense(128, activation='relu'),
#one more dropout for convergence' sake :) 
    Dropout(0.5),
#output a softmax to squash the matrix into output probabilities
    Dense(2, activation='softmax')
])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit_generator(train_batch, validation_data=valid_batch,epochs=15,steps_per_epoch=SPE ,validation_steps=VS)

model.save('models/cnnCat2.h5', overwrite=True)


# In[1]:


import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time

mixer.init()  
sound = mixer.Sound('alarm.wav')

face_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye_cascade = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

lbl=['Closed','Open'] 

model = load_model('models/cnncat2.h5')

cap = cv2.VideoCapture(0)
video = cv2.VideoCapture(r"C:\Users\mahes\Downloads\Hockey Drills.mp4")

font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
thicc=2
rpred=[99]
lpred=[99]

# Initialize current_frame to 0
current_frame = 0

# Initialize frame_skip to the number of frames you want to skip
frame_skip = 2

# Initialize a counter
counter = 0

while True:
    # Only process every frame_skip-th frame
    if counter % frame_skip == 0:
        ret, frame = cap.read() 
        ret, vid_frame = video.read()
        
        height,width = frame.shape[:2]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
        left_eye = leye_cascade.detectMultiScale(gray)
        right_eye =  reye_cascade.detectMultiScale(gray)
      
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(100,100,100),1)

        for (x,y,w,h) in right_eye:
            r_eye=frame[y:y+h,x:x+w]
            count=count+1
            r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)  
            r_eye = cv2.resize(r_eye,(24,24))
            r_eye= r_eye/255
            r_eye=r_eye.reshape(24,24,-1)
            r_eye = np.expand_dims(r_eye,axis=0)
            rpred = np.argmax(model.predict(r_eye), axis=1)
            if(rpred[0]==1):
                lbl='Open' 
            if(rpred[0]==0):
                lbl='Closed'
            break

        for (x,y,w,h) in left_eye:
            l_eye=frame[y:y+h,x:x+w]
            count=count+1
            l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
            l_eye = cv2.resize(l_eye,(24,24))
            l_eye= l_eye/255
            l_eye=l_eye.reshape(24,24,-1)
            l_eye = np.expand_dims(l_eye,axis=0)
            lpred = np.argmax(model.predict(l_eye), axis=1)
            if(lpred[0]==1):
                lbl='Open'   
            if(lpred[0]==0):
                lbl='Closed'
            break

        if(rpred[0]==0 and lpred[0]==0):
            score=score+1
            cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        else:
            score=score-1
            cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        if(score<0):
            score=0   
        cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)

        if(score<15):
            #person is feeling sleepy so we beep the alarm
            cv2.imwrite(os.path.join('image1111.jpg'),frame)

            if(thicc<16):
                thicc= thicc+2
            else:
                thicc=thicc-2
                if(thicc<2):
                    thicc=2
            cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 

            # Save the current frame number
            current_frame = video.get(cv2.CAP_PROP_POS_FRAMES)
        else:
            try:
                sound.play()
             
            except:  #isplaying = False
                pass
            # person is not feeling sleepy, play video
            video.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            video.read()

        # Create UI with face frame and video frame side-by-side  
        face_frame = cv2.resize(frame, (640,480)) 
        vid_frame = cv2.resize(vid_frame, (640,480))

        ui_frame = np.zeros((480,1280,3), np.uint8)
        ui_frame[0:480, 0:640] = face_frame 
        ui_frame[0:480, 640:1280] = vid_frame

        # Display outputs
        cv2.imshow('UI Frame',ui_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter += 1
        
cap.release()
video.release()
cv2.destroyAllWindows()


# ## The output you’re seeing is related to the execution of a machine learning model. In this context, 1/1 indicates that one batch of data is being processed out of a total of one batch. The number after the - is the time taken to process that batch. This output is typically seen when using libraries like Keras for training or predicting with neural networks. Each line represents a separate prediction made by the model. The time taken for each prediction can vary based on the complexity of the model, the size of the input data, and the hardware capabilities. So, in our case, it seems like the model is making a series of individual predictions, each on a single batch of data, and the time taken for each prediction is being printed out. 😊
