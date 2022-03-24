import sys, av, time, os
import tellopy, traceback, datetime
import cv2.cv2 as cv2
import argparse, imutils
import numpy as np, numpy

from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from PIL import Image
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing import image
import matplotlib.pyplot as plt

from os import listdir

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", default='./Datas/Models/deploy.prototxt.txt',
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model",    default='./Datas/Models/res10_300x300_ssd_iter_140000.caffemodel',
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
ap.add_argument("-s", "--save",  action='store_true',
    help="save the video")
args = vars(ap.parse_args())

def preprocess_image(image_path):
    print(image_path)
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)

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
	model.add(Activation('softmax')) # softmax #sigmoid

	#you can download pretrained weights from https://drive.google.com/file/d/1CPSeum3HpopfomUEK1gybeuIVoeJT_Eo/view?usp=sharing
	from keras.models import model_from_json
	model.load_weights('Datas/Models/vgg_face_weights.h5')

	vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)

	return vgg_face_descriptor

model = loadVggFaceModel()

#------------------------

#put your employee pictures in this path as name_of_employee.jpg
employee_pictures = "Datas/face/"

employees = dict()
idx = 0
for file in listdir(employee_pictures):
    employee, extension = file.split(".")
    employees[employee] = model.predict(preprocess_image('Datas/face/%s.jpg' % (employee)))[0,:]
print("employee representations retrieved successfully")

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))

#------------------------

def handleFileReceived(event, sender, data):
    return

if args["save"]:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi',fourcc, 20.0, (400,300))

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

def main():
    #drone = tellopy.Tello()
    landed = True
    speed = 30
    up,down,left,right,forw,back,clock,ctclock = False,False,False,False,False,False,False,False
    ai = True
    pic360 = False
    currentPic = 0
    move360 = False
    found = False
    ################################################
    cap = cv2.VideoCapture(0)                    # 0 is for /dev/video0
    ################################################

    try:
        #drone.connect()
        #drone.wait_for_connection(60.0)

        #container = av.open(drone.get_video_stream())
        #drone.subscribe(drone.EVENT_FILE_RECEIVED, handleFileReceived)
        #container = av.open("default:none")
        #frame_skip = 300
        while True:
            try:
                #for frame in container.decode(video=0):
                if True:
                    #if 0 < frame_skip:
                    #    frame_skip = frame_skip - 1
                    #    continue
                    #start_time = time.time()
                    #img = cv2.cvtColor(numpy.array(frame.to_image()), cv2.COLOR_RGB2BGR)
                    ret, img = cap.read()
                    #img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    #img=cv2.imread('person.jpg')
                    img = imutils.resize(img, width=400)
                    (h, w) = img.shape[:2]

                    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                    (300, 300), (104.0, 177.0, 123.0))
                    net.setInput(blob)
                    detections = net.forward()

                    face_dict = {}
                    copy = list()
                    for i in range(0, detections.shape[2]):  #찾은 얼굴 rect 치는 것

                        confidence = detections[0, 0, i, 2]

                        if confidence < 0.5:
                            continue

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (tstartX, tstartY, tendX, tendY) = box.astype("int")

                        cv2.rectangle(img, (tstartX, tstartY), (tendX, tendY),
                            (0, 0, 255), 2) #red


                        text = "{:.2f}%".format(confidence * 100)
                        face_dict[text]=box
                        centerX = tstartX + int( (tendX - tstartX ) / 2)
                        centerY = tstartY + int( (tendY - tstartY ) / 2)

                        if found == True:  
                            distanceX = tempCenterX - centerX
                            distanceY =  tempCenterY - centerY
                            #print("distanceX : "+str(distanceX))
                            #print("distanceY : "+str(distanceY))
                            if not( (distanceX  > -50 and distanceX < 50) and (distanceY > -50 and distanceY < 50) ):
                                #print(distanceX , distanceY)
                                continue
                            if endX - startX > tendX - tstartX + 10  :
                                continue

                        (startX, startY, endX, endY) = box.astype("int")
                        print(startX, startY, endX, endY)


                    try:
                        H,W,_ = img.shape
                        distTolerance = 0.05 * np.linalg.norm(np.array((0, 0))- np.array((w, h)))

                        box = face_dict[sorted(face_dict.keys())[0]]  #idex out of range
                        y = startY - 10 if startY - 10 > 10 else startY + 10
                        cv2.rectangle(img, (startX, startY), (endX, endY),
                            (0, 255, 0), 2) #green

                        distance = np.linalg.norm(np.array((startX,startY))-np.array((endX,endY)))
                        #print("distance : "+str(distance))

                        if found == False: #false일때 오래걸리는 이유 
                            detected_face = img[int(startY):int(endY), int(startX):int(endX)]
                            detected_face = cv2.resize(detected_face, (224, 224))

                            img_pixels = image.img_to_array(detected_face)
                            img_pixels = np.expand_dims(img_pixels, axis = 0)

                            img_pixels /= 127.5
                            img_pixels -= 1
                            captured_representation = model.predict(img_pixels)[0,:]



                            for i in employees:
                                employee_name = i

                                representation = employees[i]

                                similarity = findCosineSimilarity(representation, captured_representation)
                                if(similarity < 0.20):
                                    #drone.counter_clockwise(0)
                                    #drone.clockwise(0)
                                    #print(employee_name)
                                    #cv2.putText(img, employee_name, (int(startX+endX - startX+15), int(endY-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                                    copy.append(startX)
                                    copy.append(startY)
                                    copy.append(endX)
                                    copy.append(endY)
                                    found = True
                                    break
                        else :
                            copy.append(startX)
                            copy.append(startY)
                            copy.append(endX)
                            copy.append(endY)
                        
                        #print("hello")
                        if found == True:
                            print("True")
                            if int((startX+endX)/2) +20 < (W /2) - distTolerance :
                                #drone.counter_clockwise(30)
                                print("clockwise 30")
                                ctclock = True
                            elif int((startX+endX)/2) - 20 > (W /2) + distTolerance:
                                #drone.clockwise(30)
                                print("clockwise_______ 30")
                                clock = True
                            else:
                                if ctclock:
                                    #drone.counter_clockwise(0)
                                    print("clockwise 0")
                                    ctclock = False
                                if clock:
                                    #drone.clockwise(0)
                                    print("clockwise 0")
                                    clock = False

                            if int((startY+endY)/2) < H/2 - distTolerance :
                                #drone.up(20)
                                print("up")
                                up = True
                            elif int((startY+endY)/2) > H/2+distTolerance :
                                #drone.down(20)
                                print("down")
                                down = True
                            else:
                                if up:
                                    up = False
                                    #drone.up(0)
                                    print("up")
                                if down:
                                    down = False
                                    #drone.down(0)
                                    print("down")

                            if int(distance) < 120-distTolerance  :
                                forw = True
                                #drone.forward(20)
                                print("forward")
                            elif int(distance) > 120 + distTolerance :
                                #drone.backward(20)
                                back = True
                                print("backward")
                            else :
                                if back:
                                    back = False
                                    #drone.backward(0)
                                    print("backward")
                                if forw:
                                    forw = False
                                    #drone.forward(0)
                                    print("forward")
                        #cv2.line(img,(int((x+x+w)/2),y+15),(x+w,y-20),color,1)
                        #cv2.line(img,(x+w,y-20),(x+w+10,y-20),color,1)
                        #cv2.waitKey(1)

                        if found == True:
                            cv2.putText(img, employee_name, (int(startX+endX - startX+15), int(endY-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        #else:  #넣은면 띄엄띄엄 됨, 213번줄,"false일때 오래걸리는 이유"" 때문이다.
                        #    cv2.putText(img, "unknown", (int(startX+endX - startX+15), int(endY-12)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

                        #cv2.waitKey(0)

                    except Exception as e:
                        print(e)
                        found = False
                        if up:
                            continue
                            #drone.up(0)
                        if down:
                            continue
                            #drone.down(0)
                        if back:
                            continue
                            #drone.backward(0)
                        if forw:
                            continue
                            #drone.forward(0)

                    if args["save"]:
                            out.write(img)

                    if len(copy) == 4:
                        tempCenterX = copy[0] + int( (copy[2] - copy[0] ) / 2)
                        tempCenterY = copy[1] + int( (copy[3] - copy[1] ) / 2)

                    cv2.imshow('Original', img)

                    #if frame.time_base < 1.0/60:
                    #    time_base = 1.0/60
                    #else:
                    #    time_base = frame.time_base
                    #frame_skip = int((time.time() - start_time)/time_base)
                    keycode = cv2.waitKey(1)

                    if keycode == 32 :
                        if landed:
                            #drone.takeoff()
                            landed = False
                        else:
                            #drone.land()
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

            except Exception as e:
                print(e)
                break


    except Exception as ex:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        traceback.print_exception(exc_type, exc_value, exc_traceback)
        print(ex)
    finally:
        if args["save"]:
            out.release()
        #drone.quit()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
