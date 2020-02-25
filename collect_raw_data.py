from read_data import OBDConnection
from utils import invert_frame
import cv2
import numpy as np 
import sys, csv

file_name = sys.argv[1]
file_dir = 'data/'+file_name+'/'

resolution = (1920,1080)

cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, resolution[0])
cap.set(4, resolution[1])

out_video = cv2.VideoWriter(file_dir+file_name+'.mp4',cv2.VideoWriter_fourcc(*'XVID'), 60, resolution)

with open(file_dir+file_name+'.csv', mode='w') as csv_file:
    writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    writer.writerow(['steering_angle', 'throttle_pos', 'brake_pos', 'speed'])

    while True:
        ret, frame = cap.read()
        spd = obd.query_vehicle(obd.SPD_CODE).value

        if not ret:
            break

        vector = [None, None, None, spd]
        #print(vector)
        writer.writerow(vector)

        out_video.write(frame)
        cv2.imshow('Data Collection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()