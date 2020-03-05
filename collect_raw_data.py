from __future__ import print_function
#from read_data import OBDConnection
from panda import Panda
from utils import *
import cv2
import numpy as np 
import sys, csv, os

wheel = cv2.imread('steering_wheel.png')
wheel = cv2.resize(wheel, (200,200))
font = cv2.FONT_HERSHEY_SIMPLEX

def visualization(frame, wheel_angle, throttle_pos, brake_pos):
    frame = cv2.resize(frame, (600,300))
    visual_frame = np.zeros((700,1000,3), dtype=np.uint8)
    visual_frame[:frame.shape[0],:frame.shape[1]] = frame
    cv2.putText(visual_frame, "Steering Angle", (720,50),font,0.7,(255,255,255),2)
    wheel_cpy = rotate(wheel, wheel_angle)
    visual_frame[100:100+wheel_cpy.shape[0],700:700+wheel_cpy.shape[1]] = wheel_cpy

    textsize = cv2.getTextSize(str(wheel_angle), font, 0.7, 2)[0]

    offX = textsize[0] // 2
    offY = textsize[1] // 2

    cv2.putText(visual_frame, str(wheel_angle), (700+(wheel_cpy.shape[1]//2)-offX, 100+(wheel_cpy.shape[0]//2)+offY), font, 0.7, (0,0,0), 2)

    graph_1 = bar_graph([throttle_pos], ['Throttle Pos'], 100, (275,275))
    graph_2 = bar_graph([brake_pos], ['Brake Pos'], 100, (275,275))

    textsize = cv2.getTextSize("Throttle", font, 0.7, 2)[0]

    offX = textsize[0] // 2

    cv2.putText(visual_frame, "Throttle", (50+(graph_1.shape[1]//2),410),font,0.7,(255,255,255), 2)

    visual_frame[425:425+graph_1.shape[0],50:50+graph_1.shape[1]] = graph_1

    cv2.putText(visual_frame, "Brake", (50+graph_1.shape[1]+50+(graph_2.shape[1]//2),410),font,0.7,(255,255,255), 2)

    visual_frame[425:425+graph_2.shape[0],50+graph_1.shape[1]+50:50+graph_1.shape[1]+50+graph_2.shape[1]] = graph_2

    cv2.imshow('Data Collection', visual_frame)
    #inter_out.write(visual_frame)

'''
file_name = sys.argv[1]
file_dir = 'data/'+file_name+'/'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
'''

resolution = (1280,720)

cap = cv2.VideoCapture('tests/test_10/test_10_raw.mp4')

#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, resolution[0])
cap.set(4, resolution[1])
'''
out_video = cv2.VideoWriter(file_dir+file_name+'.mp4',cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)

try:
    print("Trying to connect to Panda over USB...")
    p = Panda()

except AssertionError:
    print("USB connection failed. Trying WiFi...")

    try:
        p = Panda("WIFI")
    except Exception as ex:
        print(ex)
        print("WiFi connection timed out. Please make sure your Panda is connected and try again.")
        sys.exit(0)
'''
try:
    '''
    outputfile = open(file_dir+file_name+'.csv', 'w')
    csvwriter = csv.writer(outputfile)
    #Write Header
    csvwriter.writerow(['Bus', 'MessageID', 'Message', 'MessageLength'])
    print("Writing csv file output.csv. Press Ctrl-C to exit...\n")

    bus0_msg_cnt = 0
    bus1_msg_cnt = 0
    bus2_msg_cnt = 0
    '''
    while True:
        ret, frame = cap.read()
        #can_recv = p.can_recv()
        #print(can_recv)
        if not ret:
            break
        '''
        
        for address, _, dat, src  in can_recv:
            csvwriter.writerow([str(src), str(hex(address)), "0x{}".format(dat.hex()), len(dat)])

            if src == 0:
                bus0_msg_cnt += 1
            elif src == 1:
                bus1_msg_cnt += 1
            elif src == 2:
                bus2_msg_cnt += 1

            print("Message Counts... Bus 0: "+str(bus0_msg_cnt)+" Bus 1: "+ str(bus1_msg_cnt)+" Bus 2: "+str(bus2_msg_cnt), end='\r')
            '''
        #out_video.write(frame)
        visualization(frame, 40, 20, 10)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
except KeyboardInterrupt:
    '''
    print("\nNow exiting. Final message Counts... Bus 0: "+str(bus0_msg_cnt)+" Bus 1: "+ str(bus1_msg_cnt)+" Bus 2: "+str(bus2_msg_cnt))
    outputfile.close()
    '''
    #cap.release()
    #cv2.destroyAllWindows()

if __name__ == "__main__":
    can_logger()
