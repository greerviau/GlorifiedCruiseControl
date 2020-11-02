from utils.visual_utils import *
from selfdriving.car.toyota.interface import Interface
from datetime import datetime
import numpy as np 
import cv2, sys, csv, os

RESOLUTION = (1280, 720)
FULLSCREEN = False
FPS = 30
COUNT = 0
COLLECT_FPS = 1 #capture every frame

try:
    #Create CAN interface
    interface = Interface()

    #Create video capture and set hyperparams
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, False)
    cap.set(cv2.CAP_PROP_FPS, FPS)
    cap.set(3, RESOLUTION[0])
    cap.set(4, RESOLUTION[1])

    #Create data dir
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
        raise Exception('Cannot Override Existing Folder')
    
    #Create csv file
    outputfile = open(os.path.join(file_dir,file_name)+'.csv', 'w')
    csvwriter = csv.writer(outputfile)
    #Write Header
    csvwriter.writerow(['Frame Id','SAS Raw Hex', 'Steering Angle', 'Steering Torque', 'Gas Raw Hex', 'Gas Pedal Pos', 'Brake Raw Hex', 'Brake Pedal Pos', 'Speed Raw Hex', 'Speed'])
    print('Writing csv file {}.csv. Press Ctrl-C to exit...\n'.format(file_name))
    
    while True:
        
        ret, frame = cap.read()
        if not ret:
            raise Exception('No Frame Detected')
        
        frame = invert_frame(frame)

        sas_raw, sas_angle, sas_torque, accel_raw, accel_pos, brake_raw, brake_pos, speed_raw, speed = interface.get_can_messages()
        COUNT+=1

        visualization(frame.copy(), sas_angle, accel_pos, brake_pos, speed, fullscreen=FULLSCREEN)

        #Capture every N frames
        if COUNT % COLLECT_FPS == 0:
            frame_id = 'frame_{}.png'.format(COUNT)
            cv2.imwrite(os.path.join(image_dir, frame_id), cv2.resize(frame, (640, 360)))
            csvwriter.writerow([frame_id, sas_raw, sas_angle, sas_torque, accel_raw, accel_pos, brake_raw, brake_pos, speed_raw, speed])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
                    
except Exception as ex:
    print(ex)
    print('Exiting')
    outputfile.close()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)