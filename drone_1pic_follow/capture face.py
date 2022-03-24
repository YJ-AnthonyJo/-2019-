import cv2, sys
import time
import datetime

if __name__=="__main__":
    cap = cv2.VideoCapture(0)                    # 0 is for /dev/video0
    while True:
        ret, frm = cap.read()
        
        dst = frm.copy() 
        #dst = frm[100:324, 200:424]


        if ret == True:
            #cv2.imshow('frm', frm)
            cv2.imshow('a', dst)
        key = cv2.waitKey(1);

        if key == ord('q'):
            break
        elif key == ord('s'):
            file = 'person.jpg'
            cv2.imwrite(file,dst)
            print(file, ' saved')
    cap.release()
    cv2.destroyAllWindows()