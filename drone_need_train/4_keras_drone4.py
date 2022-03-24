# --------------------------------------------------------------------------------------------------------------------------
# Tello Keras Drone
# - Keras Vision Inference on Tello EDU Drone
#
# 03/01/2019, Espace (ITE College West)
# --------------------------------------------------------------------------------------------------------------------------
#!/usr/bin/env python3
import threading
import socket
import sys
import time

import cv2
import numpy as np
from keras.models import model_from_json

# Tello IP Control Variables
# ----------------------------------------------
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
host = ''
port = 9000
locaddr = (host,port)

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# constants for drones list
DRONES_IP_ADDRESS   = 0
DRONES_IP_PORT      = 1
DRONES_TIMEOUT      = 2
DRONES_OK_STATUS    = 3
DRONES_CURR_CMD     = 4
DRONES_CURR_TIMEOUT = 5

# drones list
drones = [
    # IP address       port, timeout, ok status
    [ '192.168.10.1', 8889, 0, 0, '', 0 ],    # drone 1
                                              # drone 2
                                              # drone 3
                                              # ...
                                              # drone n
]

sock.bind(locaddr)

# --------------------------------------------------------------------------------------------------------------------------
# socket recv handler.  Its run on a seperate thread and keeps listening.
# --------------------------------------------------------------------------------------------------------------------------
def recv():
    print ('\nReceive Tread started.\n')
    while True:
        try:
            data, server = sock.recvfrom(1518)

            # if OK status received, stored the status to drone which have same IP address
            for j in range(len(drones)):
                if server[0] == drones[j][DRONES_IP_ADDRESS] and data.decode(encoding="utf-8")=='ok':
                    drones[j][DRONES_OK_STATUS] = 1

                    # Once valid response is received by one drone, make expect rest of the drones response to be soon (i.e. timeout 500msec sooner)
                    curr_milli_sec = int(round(time.time() * 1000))
                    for i in range(len(drones)):
                        drones[i][DRONES_TIMEOUT] = curr_milli_sec + 500

        except Exception:
            print ('\nReceive Thread Exception.\n')
            break

# --------------------------------------------------------------------------------------------------------------------------
# Send commands to 2 Tello EDU and checks whether it execute the commands by looking for the 'ok' reply from each IP address.
# If not, the command at timeout will re-send to the drone.
#
# Only for commands that reply OK status.
#
# cmd - the command to do
# timeout - the timeout in milli seconds.
# --------------------------------------------------------------------------------------------------------------------------
def sendCommand2AllNEnsureDo(cmd, timeout):

    cmd = cmd.encode(encoding="utf-8")

    # Reset all OK status and force timeout: ensure commands are send out at 1st entry of the while loop
    for i in range(len(drones)):
        drones[i][DRONES_OK_STATUS] = 0
        drones[i][DRONES_TIMEOUT] = 0

    while True:

        # Send Command to Drones which have not receive OK and did timeout
        current_milli_sec = int(round(time.time() * 1000))
        for i in range(len(drones)):
            if drones[i][DRONES_OK_STATUS] == 0 and drones[i][DRONES_TIMEOUT] < current_milli_sec:
                tello_address = (drones[i][DRONES_IP_ADDRESS], drones[i][DRONES_IP_PORT])
                sent = sock.sendto(cmd, tello_address)
                drones[i][DRONES_TIMEOUT] = current_milli_sec + timeout

        # Check whether all drones receive OK status
        exit_count = 0
        for k in range(len(drones)):
            if drones[k][DRONES_OK_STATUS] == 1:
                exit_count = exit_count + 1

        if exit_count >= len(drones): # break if all drones receive OK status
            break

    return

# --------------------------------------------------------------------------------------------------------------------------
# Send commands to 2 Tello EDU and checks whether it execute the commands by looking for the 'ok' reply from each IP address.
# If not, the command at timeout will re-send to the drone.
#
# Only for commands that reply OK status.
#
# cmd - the command to do
# timeout - the timeout in milli seconds.
# --------------------------------------------------------------------------------------------------------------------------
def sendCommand2All(cmd, timeout):

    cmd = cmd.encode(encoding="utf-8")

    # Reset all OK status and force timeout: ensure commands are send out at 1st entry of the while loop
    for i in range(len(drones)):
        drones[i][DRONES_OK_STATUS] = 0
        drones[i][DRONES_TIMEOUT] = 0


    # Send Command to Drones which have not receive OK and did timeout
    current_milli_sec = int(round(time.time() * 1000))
    for i in range(len(drones)):
        if drones[i][DRONES_OK_STATUS] == 0 and drones[i][DRONES_TIMEOUT] < current_milli_sec:
            tello_address = (drones[i][DRONES_IP_ADDRESS], drones[i][DRONES_IP_PORT])
            sent = sock.sendto(cmd, tello_address)
            drones[i][DRONES_CURR_CMD] = cmd
            drones[i][DRONES_CURR_TIMEOUT] = timeout
            drones[i][DRONES_TIMEOUT] = current_milli_sec + timeout

    return

def CheckAllDo():

    # Send Command to Drones which have not receive OK and did timeout
    current_milli_sec = int(round(time.time() * 1000))
    for i in range(len(drones)):
        if drones[i][DRONES_OK_STATUS] == 0 and drones[i][DRONES_TIMEOUT] < current_milli_sec:
            tello_address = (drones[i][DRONES_IP_ADDRESS], drones[i][DRONES_IP_PORT])
            sent = sock.sendto(drones[i][DRONES_CURR_CMD], tello_address)
            drones[i][DRONES_TIMEOUT] = current_milli_sec + drones[i][DRONES_CURR_TIMEOUT]
            print ('Resending...\r\n')

    exit_count = 0

    for k in range(len(drones)):
        if drones[k][DRONES_OK_STATUS] == 1:
            exit_count = exit_count + 1

    if exit_count >= len(drones): # break if all drones receive OK status
        nRet = 1
    else:
        nRet = 0

    return nRet

def main():
    # print instructions for user
    # ---------------------------
    print ('\r\n\r\nTello Keras Drone\r\n')

    # recvThread create
    recvThread = threading.Thread(target=recv)
    recvThread.start()

    # Loading model
    with open('DataforAI\\model.json', 'r') as file_model:
        model_desc = file_model.read()
        model = model_from_json(model_desc)

    model.load_weights('DataforAI\\weights.h5')

    #sendCommand2AllNEnsureDo('command', 500)
    #sendCommand2AllNEnsureDo('streamon', 500)
    #sendCommand2AllNEnsureDo('mon', 1000)
    #sendCommand2AllNEnsureDo('mdirection 0', 500)

    # Open the video source
    # if it is video source is file
    # video_dev = cv2.VideoCapture('c:/testAI/PetImages/CatDogVideos/Puppy1047.mp4')
    #video_dev = cv2.VideoCapture('udp://192.168.10.1:11111')
    #video_width = video_dev.get(cv2.CAP_PROP_FRAME_WIDTH)
    #video_height = video_dev.get(cv2.CAP_PROP_FRAME_HEIGHT)




    # if it is video source is camera
    video_dev = cv2.VideoCapture(0)


    drone_complete_action = 0
    flyforward = 0
    mission_pad_alignment = 0
    mission_pad_number = 1

    # Start Flight
    print ('Tello Flying off\r\n')
    #sendCommand2AllNEnsureDo('takeoff', 4000)

    #drone_complete_action = CheckAllDo()
    # Main loop
    try:
        prev_timestamp = time.time()
        prev_class_id = 7

        while True:
            ret, orig_image = video_dev.read()
            curr_time = time.localtime()
            bef=orig_image
######################################################################################33
            height, width = orig_image.shape[:2]
            #print("0")
            
            img_color = cv2.resize(orig_image, (width, height), interpolation=cv2.INTER_AREA)
            #print("1")

            # 원본 영상을 HSV 영상으로 변환합니다.
            img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

            # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
            lower_blue1 = np.array([106,  83,  83])
            upper_blue1 = np.array([116, 255, 255])

            lower_blue2 = np.array([96, 83, 83])
            upper_blue2 = np.array([106, 255, 255])
            
            lower_blue3 = np.array([96, 83, 83])
            upper_blue3 = np.array([106, 255, 255])
            ######
            img_mask1 = cv2.inRange(img_hsv, lower_blue1, upper_blue1)
            img_mask2 = cv2.inRange(img_hsv, lower_blue2, upper_blue2)
            img_mask3 = cv2.inRange(img_hsv, lower_blue3, upper_blue3)
            img_mask = img_mask1 | img_mask2 | img_mask3

            kernel = np.ones((5,5),np.uint8)
            img_mask= cv2.morphologyEx(img_mask,cv2.MORPH_OPEN,kernel)
            img_mask= cv2.morphologyEx(img_mask,cv2.MORPH_CLOSE,kernel)
            #print('2')

            # 마스크 이미지로 원본 이미지에서 범위값에 해당되는 영상 부분을 획득합니다.
            img_result = cv2.bitwise_and(img_color, img_color, mask=img_mask)
            numOfLabels, img_label,stats,centroids =cv2.connectedComponentsWithStats(img_mask)
            pp=0
            for idx, centroid in enumerate(centroids):
                pp=1
                if stats[idx][0] == 0 and stats[idx][1] ==0:
                    continue
                if np.any(np.isnan(centroid)):
                    continue
                x,y,width,height,area=stats[idx]
                centerX, cneterY= int(centroid[0]), int(centroid[1])

                if area>20000:
                    cv2.rectangle(orig_image,(x,y),(x+width,y+height),(0,0,255))
                    orig_image=orig_image[y:y+height ,x:x+width]
########################################################################################
                    input_width = 48
                    input_height = 48
                    resized_image = cv2.resize(
                        orig_image,
                        (input_width, input_height),
                    ).astype(np.float32)
                    normalized_image = resized_image / 255.0

                    # Execution forecast
                    batch = normalized_image.reshape(1, input_height, input_width, 3)
                    result_onehot = model.predict(batch)
                    #left_score, right_score, land_score, forward_score, other_score = result_onehot[0]
                    class_id = np.argmax(result_onehot, axis=1)[0]

                    if class_id == 0:
                        class_str = 'left'
                    elif class_id == 1:
                        class_str = 'right'
                    elif class_id == 2:
                        class_str = 'land'
                    elif class_id == 3:
                        class_str = 'forward'
                    elif class_id == 4:
                        class_str = 'backward'
                    elif class_id == 5:
                        class_str = 'other'
                    print(class_id, result_onehot[0])
                    print(class_str)

                    # Command to Drone after inference
                    """if flyforward == 0: #and mission_pad_alignment == 0:
                        if drone_complete_action == 1:
                            if class_id == 0 and left_score >0.90:       # left
                                print ('Fly Left\r\n')
                                sendCommand2All('left 60', 3000)
                                drone_complete_action = 0
                                mission_pad_alignment = 1
                            elif class_id == 1 and right_score > 0.90:     # right
                                print ('Fly Right\r\n')
                                sendCommand2All('right 60', 3000)
                                drone_complete_action = 0
                                mission_pad_alignment = 1
                            elif class_id == 2 and land_score >0.90:     # land
                                print ('Landing...\r\n')
                                sendCommand2All('land', 4000)
                                drone_complete_action = 0
                            elif class_id == 3 and forward_score > 0.90:     # forward
                                #class_str = 'forward'
                                print ('fwd\r\n')
                                sendCommand2All('forward 60', 4000)
                                drone_complete_action = 0
                                #flyforward = 1
                            elif class_id == 4 and other_score > 0.90:     # other
                                class_str = 'other'
                    elif flyforward > 0:
                        if drone_complete_action == 1:
                            if flyforward == 1:
                                sendCommand2All('forward 50', 4000)
                                print ('forward\r\n')
                                drone_complete_action = 0
                                flyforward = 2
                            elif flyforward == 2:
                                sendCommand2All('right 30', 4000)
                                print ('fwd - right\r\n')
                                drone_complete_action = 0
                                flyforward = 3
                            elif flyforward == 3:
                                m_cmd = 'go 0 0 80 20 m' + str(mission_pad_number)
                                mission_pad_number = mission_pad_number + 1
                                if mission_pad_number > 8:
                                    mission_pad_number = 1
                                sendCommand2All(m_cmd, 2000)
                                print ('fwd - align\r\n')
                                drone_complete_action = 0
                                flyforward = 0
                    elif mission_pad_alignment > 0:
                        if drone_complete_action == 1:
                            m_cmd = 'go 0 0 80 20 m' + str(mission_pad_number)
                            mission_pad_number = mission_pad_number + 1
                            if mission_pad_number > 8:
                                mission_pad_number = 1
                            sendCommand2All(m_cmd, 2000)
                            print ('align mission pad\r\n')
                            drone_complete_action = 0
                            mission_pad_alignment = 0

                    drone_complete_action = CheckAllDo()
                    """
                    prev_class_id = class_id

                    # Calculate execution time
                    recent_timestamp = time.time()
                    period = recent_timestamp - prev_timestamp
                    prev_timestamp = recent_timestamp

                    line1 = ('time=%02d:%02d:%02d ' % (curr_time.tm_hour, curr_time.tm_min, curr_time.tm_sec))
                    #line2 = ('cat=：%.2f %.2f %.2f %.2f %.2f' % (left_score, right_score, land_score, forward_score, other_score))
                    line3 = ('category= %s' % class_str)
                    line4 = ('Time Elapse %f' % period)

                    # display image
                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    TopLeftCornerOfText1 = (5,40)
                    TopLeftCornerOfText2 = (5,80)
                    TopLeftCornerOfText3 = (5,120)
                    TopLeftCornerOfText4 = (5,160)
                    fontScale              = 1
                    fontColor              = (255,255,255)
                    lineType               = 2

                    cv2.putText(bef, line1, TopLeftCornerOfText1, font, fontScale, fontColor, lineType)
                    #cv2.putText(orig_image, line2, TopLeftCornerOfText2, font, fontScale, fontColor, lineType)
                    cv2.putText(bef, line3, TopLeftCornerOfText3, font, fontScale, fontColor, lineType)
                    cv2.putText(bef, line4, TopLeftCornerOfText4, font, fontScale, fontColor, lineType)
                    cv2.imshow('video', orig_image)
                    key = cv2.waitKey(1);

                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Start Flight
                        print ('User: Tello Flying off\r\n')
                        sendCommand2All('takeoff', 7000)
                        mission_pad_alignment = 1
                        drone_complete_action = 0
                        mission_pad_number = 1
                    elif key == ord('d'):
                        print ('User landing\r\n')
                        sendCommand2All('land', 7000)
                        drone_complete_action = 0
                    elif key == ord('l'):
                        print ('User left\r\n')
                        sendCommand2All('left 20', 4000)
                        drone_complete_action = 0
                    elif key == ord('r'):
                        print ('User right\r\n')
                        sendCommand2All('right 20', 4000)
                        drone_complete_action = 0
                    elif key == ord(','):
                        print ('User ccw\r\n')
                        sendCommand2All('ccw 5', 4000)
                        drone_complete_action = 0
                    elif key == ord('.'):
                        print ('User cw\r\n')
                        sendCommand2All('cw 5', 4000)
                        drone_complete_action = 0
                    elif key == ord('b'):
                        print ('User back\r\n')
                        sendCommand2All('back 20', 4000)
                        drone_complete_action = 0
                    elif key == ord('f'):
                        print ('User forward\r\n')
                        sendCommand2All('forward 20', 4000)
                        drone_complete_action = 0
                if pp==1:
                    pp=0
                    break

            #########################################수정 필요#######################################3
            #gray = cv2.cvtColor(orig_image, cv2.COLOR_BGR2GRAY)
            #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(30, 30))
            #########################################수정 필요#######################################3
            
            #for (x,y,w,h) in faces:
                 # print (x,y,w,h)
            #     color = (255,0,0) #BGR color
            #     stroke = 2
            #     end_coord_x = x + w
            #     end_coord_y = y + h
            #     cv2.rectangle (orig_image,(x, y),(end_coord_x, end_coord_y), color, stroke)
            cv2.imshow('bef', bef)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Check if the stream is over
            if ret is None or orig_image is None:
                break

            # Scale to the dimension entered by the model, adjust the value range from 0 to 1.
            

    except KeyboardInterrupt:
        print('User interruption')

    # End image device
    #sendCommand2All('land', 3000)
    video_dev.release()
    cv2.destroyAllWindows()
    #sock.close()


if __name__ == '__main__':
    main()
