"""
videoCapture.py
"""
import cv2, sys
import time
import datetime
import socket

tello_address = ('192.168.10.1', 8889)
local_address = ('', 9000)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(local_address)

def send(message):
  try:
    sock.sendto(message.encode(), tello_address)
    print("Sending message: " + message)
  except Exception as e:
    print("Error sending: " + str(e))

if __name__=="__main__":
    #send("command")
    #send("streamon")
    if sys.platform == "win32":
        import os, msvcrt
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
        cap = cv2.VideoCapture(0)                    # 0 is for /dev/video0
    st=time.time()
    while True:
        if time.time()-st >=10:
            st=time.time()
            #send("command")
        ret, frm = cap.read()
        if ret == True:
            cv2.imshow('frm', frm)

        key = cv2.waitKey(1);

        if key == ord('q'):
            break
        elif key == ord('f'): #쎈
            file = "DataforAI/forward/"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
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
        elif key == ord('z'): #left Turn
            file = "DataforAI\\lturn\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,dst)
            print(file, ' saved')
        elif key == ord('x'): #Right Turn
            file = "DataforAI\\rturn\\"+datetime.datetime.now().strftime("%Y%m%d_%H%M%S%f") +'.jpg'
            cv2.imwrite(file,dst)
            print(file, ' saved')    

    cap.release()
    cv2.destroyAllWindows()