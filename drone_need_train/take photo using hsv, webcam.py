"""
videoCapture.py
"""
import cv2, sys
import time
import datetime
import socket
import numpy as np
#tello_address = ('192.168.10.1', 8889)#
#local_address = ('', 9000)
#sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.bind(local_address)

def send(message):
  try:
    sock.sendto(message.encode(), tello_address)
    print("Sending message: " + message)
  except Exception as e:
    print("Error sending: " + str(e))


if __name__=="__main__":
    cap = cv2.VideoCapture(0)                    # 0 is for /dev/video0
    while True:
        ret, frm = cap.read()
        #dst = frm.copy() 
        #dst = frm[0:720, 0:720]

        height, width = frm.shape[:2]
        #print("0")
        
        img_color = cv2.resize(frm, (width, height), interpolation=cv2.INTER_AREA)
        #print("1")

        # 원본 영상을 HSV 영상으로 변환합니다.
        img_hsv = cv2.cvtColor(img_color, cv2.COLOR_BGR2HSV)

        # 범위 값으로 HSV 이미지에서 마스크를 생성합니다.
        lower_blue1 = np.array([107,  64,  64]) 
        upper_blue1 = np.array([117, 255, 255])

        lower_blue2 = np.array([97, 64, 64])
        upper_blue2 = np.array([107, 255, 255])
        
        lower_blue3 = np.array([97, 64, 64])
        upper_blue3 = np.array([107, 255, 255])
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
        flag=0
        for idx, centroid in enumerate(centroids):
            flag=1
            if stats[idx][0] == 0 and stats[idx][1] ==0:
                continue
            if np.any(np.isnan(centroid)):
                continue
            x,y,width,height,area=stats[idx]
            centerX, cneterY= int(centroid[0]), int(centroid[1])

            if area>20000:
                #cv2.rectangle(frm,(x,y),(x+width,y+height),(0,0,255))
                frm=frm[y:y+height ,x:x+width]
            if flag==1:
                break
                ##############################3
        if ret == True:
            cv2.imshow('frm', frm)
            #cv2.imshow('a', dst)
        key = cv2.waitKey(1);


        if key == ord('q'):
            break
        elif key == ord('f'): #쎈
            file = "DataforAI\\forward\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('b'): # 필통
            file = "DataforAI\\backward\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('l'): # 왼손
            file = "DataforAI\\left\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('r'): # 오른손 
            file = "DataforAI\\right\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('t'): #takeoff 음악책
            file = "DataforAI\\takeoff\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('e'):  #####land!!!! 노트 
            file = "DataforAI\\land\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('p'): 
            file = "DataforAI\\test\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('o'): #배경
            file = "DataforAI\\other\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('x'): #left Turn
            file = "DataforAI\\lturn\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')
        elif key == ord('z'): #Right Turn
            file = "DataforAI\\rturn\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,frm)
            print(file, ' saved')    

    cap.release()
    cv2.destroyAllWindows()