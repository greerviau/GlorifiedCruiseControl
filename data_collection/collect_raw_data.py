from utils.visual_utils import *
from selfdriving.car.toyota.interface import Interface
import cv2
from datetime import datetime
import numpy as np 
import sys, csv, os

RESOLUTION = (1280, 720)
FULLSCREEN = False
FPS = 30
COUNT = 0
COLLECT_FPS = 1 #capture every frame

try:
    interface = Interface()

    cap = cv2.VideoCapture('/dev/video2')
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(3, RESOLUTION[0])
    cap.set(4, RESOLUTION[1])

    today_date = str(datetime.date(datetime.now()))

    data_dir = '/home/greer/Documents/GCC_Data/4Runner'
    drives = os.listdir(data_dir)
    todays_drives = [d for d in drives if today_date in d]

    file_name = today_date+'_drive_{}'.format(len(todays_drives)+1)
    file_dir = os.path.join(data_dir, file_name)
    image_dir = os.path.join(file_dir, 'images')
    
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
        os.makedirs(image_dir)
    else:
        print('Cannot Override Existing Folder')
        print('Exiting')
        raise KeyboardInterrupt
    
    outputfile = open(os.path.join(file_dir,file_name)+'.csv', 'w')
    csvwriter = csv.writer(outputfile)
    #Write Header
    csvwriter.writerow(['Frame Id','SAS Raw Hex', 'Steering Angle', 'Steering Torque', 'Gas Raw Hex', 'Gas Pedal Pos', 'Brake Raw Hex', 'Brake Pedal Pos', 'Speed Raw Hex', 'Speed'])
    print('Writing csv file {}.csv. Press Ctrl-C to exit...\n'.format(file_name))
    
    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            print('No Frame Detected')
            raise KeyboardInterrupt
        
        frame = invert_frame(frame)

        sas_raw, sas_angle, sas_torque, accel_raw, accel_pos, brake_raw, brake_pos, speed_raw, speed = interface.get_can_messages()
        COUNT+=1

        visualization(frame, sas_angle, accel_pos, brake_pos, speed, fullscreen=FULLSCREEN)

        if COUNT % COLLECT_FPS == 0:
            frame_id = 'frame_{}.png'.format(COUNT)
            cv2.imwrite(os.path.join(image_dir, frame_id), frame)
            csvwriter.writerow([frame_id, sas_raw, sas_angle, sas_torque, accel_raw, accel_pos, brake_raw, brake_pos, speed_raw, speed])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
                    
except KeyboardInterrupt:
    
    print('Exiting')
    outputfile.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)