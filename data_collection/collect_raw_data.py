from utils.visual_utils import *
from selfdriving.car.toyota_4runner_2018.interface import Interface
import cv2
import numpy as np 
import sys, csv, os

RESOLUTION = (1280, 720)
FPS = 30
COUNT = 0
COLLECT_FPS = 10

interface = Interface()

cap = cv2.VideoCapture('/dev/video2')
cap.set(cv2.CAP_PROP_FPS, FPS)
cap.set(3, RESOLUTION[0])
cap.set(4, RESOLUTION[1])

file_name = sys.argv[1]
file_dir = '/home/greer/Documents/GCC_Data/4Runner/'+file_name
image_dir = os.path.join(file_dir, 'images')
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
    os.makedirs(image_dir)
else:
    print('Cannot Override Existing Folder')
    print('Exiting')
    sys.exit(0)

try:
    
    outputfile = open(os.path.join(file_dir,file_name)+'.csv', 'w')
    csvwriter = csv.writer(outputfile)
    #Write Header
    csvwriter.writerow(['Frame Id','Steering Angle','Gas Pedal Pos','Brake Pedal Pos', 'Speed'])
    print('Writing csv file {}.csv. Press Ctrl-C to exit...\n'.format(file_name))
    
    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            print('No Frame Detected')
            raise KeyboardInterrupt
        
        frame = invert_frame(frame)

        sas_angle, accel_pos, brake_pos, speed = interface.get_can_messages()
        COUNT+=1

        visualization(frame, sas_angle, accel_pos, brake_pos, speed)

        if COUNT % COLLECT_FPS == 0:
            frame_id = 'frame_{}.png'.format(COUNT)
            cv2.imwrite(os.path.join(image_dir, frame_id), frame)
            csvwriter.writerow([frame_id, sas_angle, accel_pos, brake_pos, speed])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
                    
except KeyboardInterrupt:
    
    print('Exiting')
    outputfile.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)