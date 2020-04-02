import numpy as np 
import cv2, os
import time

#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
#cap.set(3, 1280)
#cap.set(4, 720)
#print(cap.get(3))
#print(cap.get(4))
cap = cv2.VideoCapture('tests/test_06/test_06_raw.mp4')

i = 1

if not os.path.exists('test_images'):
    os.makedirs('test_images')

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('cap')
        cv2.imwrite('test_images/test_{}.jpg'.format(i),frame)
        i+=1
    time.sleep(0.1)

cap.release()
cv2.destroyAllWindows()