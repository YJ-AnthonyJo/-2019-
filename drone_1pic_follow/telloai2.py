#얼굴 보이면 따라가는 코드
from multiprocessing import Process, Queue
import sys
import traceback
import tellopy
import av
import cv2.cv2 as cv2  # for avoidance of pylint error
#import socket
import time
import os
import datetime
import imutils
import numpy as np
import argparse

from absl import logging
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

import logging

from os import listdir
#-----------------------

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='./resnet10/deploy.prototxt.txt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",    default='./resnet10/res10_300x300_ssd_iter_140000.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save",  action='store_true',
    help="save the video")
args = vars(ap.parse_args())

tello_address = ('192.168.10.1', 8889)
local_address = ('192.168.10.2', 9000)
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.bind(local_address)


def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

    #preprocess_input normalizes input in scale of [-1, +1]. You must apply same normalization in prediction.
    #Ref: https://github.com/keras-team/keras-applications/blob/master/keras_applications/imagenet_utils.py (Line 45)
    img = preprocess_input(img)
    return img

def loadVggFaceModel():
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))

    #you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
    from keras.models import model_from_json
    model.load_weights('Datas\\Models\\vgg_face_weights.h5')

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

    return vgg_face_descriptor

def send(message):
  try:
    sock.sendto(message.encode(), tello_address)
    print("Sending message: " + message)
  except Exception as e:
    print("Error sending: " + str(e))


def handleFileReceived(event, sender, data):
    global date_fmt
    # Create a file in ~/Pictures/ to receive image data from the drone.
    path = '%s/tello-%s.jpeg' % (
        os.getenv('HOMEPATH'),                              #Changed from Home to Homepath
        datetime.datetime.now().strftime('%Y-%m-%d_%H%M%S'))
    with open(path, 'wb') as fd:
        fd.write(data)
    #print('Saved photo to ',path)

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))


color = (67,67,67)

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

#def main():
    
    #cv2.destroyAllWindows()
    #p1.join()
    #p2.join()
def show():
    global ret
    #global img
    #time.sleep(2)
    print("aaasdfsdfsdfsdfsfdfdsf")
    while True:
        #ret, img = cap.read()
        print("in")
        img = q.get()
        cv2.imshow('Original',img)
if __name__ == '__main__':
    #global cap
    #global img
    
    p1 = Process(target=show)
    q=Queue()
    model = loadVggFaceModel()
    #send("command")
    #send("streamon")
    #------------------------

    #put your employee pictures in this path as name_of_employee.jpg
    employee_pictures = "Datas\\face"

    employees = dict()

    for file in listdir(employee_pictures):
        employee, extension = file.split(".")
        employees[employee] = model.predict(preprocess_image('Datas\\face\\%s.jpg' % (employee)))[0,:]

    print("employee representations retrieved successfully")


    #drone = tellopy.Tello()
    print("aaaaaa")
    #cap = cv2.VideoCapture("udp://192.168.10.1:11111")
    cap = cv2.VideoCapture(0)
    p1.start()
    print("bbbbbb")
    landed = True
    speed = 30
    up,down,left,right,forw,back,clock,ctclock = False,False,False,False,False,False,False,False
    ai = True
    pic360 = False
    currentPic = 0
    move360 = False
    #try:
    print('here')
    if True:
        #ret, img = cap.read()
        flag=0
        while True:
            #print("i")
            #if flag==1:
            #    print("p1 start")
            #    p1.start()
            #    flag=2
            if True:
                if True:
                    #if flag != 2:
                    print("1")
                    ret, img= cap.read()
                    print("2")
                    #if flag == 2:
                    q.put(img)
                    print('3')
                    #cv2.imshow('aa',img)
                    img = imutils.resize(img, width=400)
                    faces = face_cascade.detectMultiScale(img, 1.3, 5)
                    for (x,y,w,h) in faces:
                        print(x,y,w+x,h+y) 
                        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)#blue #draw rectangle to main image
                        
                        detected_face = img[int(y):int(y+h), int(x):int(x+w)] #crop detected face
                        detected_face = cv2.resize(detected_face, (224, 224)) #resize to 224x224

                        img_pixels = image.img_to_array(detected_face)
                        img_pixels = np.expand_dims(img_pixels, axis = 0)
                        #img_pixels /= 255
                        #employee dictionary is using preprocess_image and it normalizes in scale of [-1, +1]
                        img_pixels /= 127.5
                        img_pixels -= 1

                        captured_representation = model.predict(img_pixels)[0,:]

                        found = 0
                        for i in employees:
                            employee_name = i
                            representation = employees[i]

                            similarity = findCosineSimilarity(representation, captured_representation)
                            if(similarity < 0.30):
                                cv2.putText(img, employee_name, (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                                found = 1
                                break
                        
                        if(found == 0): #if found image is not in employee database
                            cv2.putText(img, 'unknown', (int(x+w+15), int(y-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                            continue
                        startX=x
                        startY=y
                        endX=x+w
                        endY=y+h
                        #try:
                        if found == 1:    
                            #print("aa")
                            H,W,_ = img.shape
                            distTolerance = 0.05 * np.linalg.norm(np.array((0, 0))- np.array((w, h)))

                            #box = face_dict[sorted(face_dict.keys())[0]]
                            y = startY - 10 if startY - 10 > 10 else startY + 10
                            cv2.rectangle(img, (startX, startY), (endX, endY),
                                (0, 255, 0), 2) #red

                            distance = np.linalg.norm(np.array((startX,startY))-np.array((endX,endY)))
                            #print("bb")
                            if int((startX+endX)/2) < W/2-distTolerance :
                                print('CounterClock')
                                #drone.counter_clockwise(30)
                                #send('cw 30')
                                ctclock = True
                            elif int((startX+endX)/2) > W/2+distTolerance:
                                print('Clock')
                                #send('ccw 30')
                                #drone.clockwise(30)
                                clock = True
                            else:
                                if ctclock:
                                    #drone.counter_clockwise(0)
                                    ctclock = False
                                    print('CTClock 0')
                                if clock:
                                    #drone.clockwise(0)
                                    clock = False
                                    print('Clock 0')
                            
                            if int((startY+endY)/2) < H/2-distTolerance :
                                #drone.up(30)
                                print('Up')
                                #send("up 30")
                                up = True
                            elif int((startY+endY)/2) > H/2+distTolerance :
                                #drone.down(30)
                                print('Down')
                                #send('down 30')
                                down = True
                            else:
                                if up:
                                    up = False
                                    print('Up 0')
                                    #drone.up(0)

                                if down:
                                    down = False
                                    print('Down 0')
                                    #drone.down(0)

                            print(int(distance))

                            if int(distance) < 110-distTolerance  :
                                forw = True
                                print('Forward')
                               #send("forward 30")
                                #drone.forward(30)
                            elif int(distance) > 110+distTolerance :
                                #drone.backward(30)
                                print('Backward00')
                                #send('back 30')
                                back = True
                            else :
                                if back:
                                    back = False
                                    print('Backward 0')
                                    #drone.backward(0)
                                if forw:
                                    forw = False
                                    print('Forward 0')
                                    #drone.forward(0)
                                

                        #except Exception as e:
                            #print(e)
                        #    None

                    
                    #cv2.imshow('Original', img)
                    #print("here1")
                    if flag==0:
                        flag=1
                    keycode = cv2.waitKey(1)
                    if keycode == 32 :
                        if landed:
                            #drone.takeoff()
                            #send('takeoff')
                            landed = False
                        else:
                            #drone.land()
                            #send('land')
                            landed = True

                    if keycode == 27 :
                        raise Exception('Quit')

                    if keycode == 13 :
                        #drone.take_picture()
                        time.sleep(0.25)
                        #pic360 = True
                        #move360 = True

                    if keycode & 0xFF == ord('q') :
                        pic360 = False
                        move360 = False 
    cv2.destroyAllWindows()
    p1.join()