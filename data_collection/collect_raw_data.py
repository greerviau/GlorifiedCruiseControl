from __future__ import print_function, unicode_literals
from panda import Panda
from GlorifiedCruiseControl.utils.utils import *
import cv2
import numpy as np 
import sys, csv, os

'''
PIDS AND SLICES MUST BE ADJUSTED
FOR DIFFERENT VEHICLES
//////////////////////////////////////
'''

#PIDS FOR SENSORS
sas_pin = '0x25'
gas_pin = '0x2c1'
brake_pin = '0x224'
speed_pin = '0xb4'

#MESSAGE SLICES
sas_slice = slice(0,2)
gas_slice = slice(6,7)
brake_slice = slice(4,6)
speed_slice = slice(5,7)

################################

file_name = sys.argv[1]
file_dir = '../GCC_Data/4Runner/'+file_name+'/'
if not os.path.exists(file_dir):
    os.makedirs(file_dir)
else:
    print('Cannot Override Existing Folder')
    print('Exiting')
    sys.exit(0)

resolution = (640, 360)
fps = 30

#cap = cv2.VideoCapture('tests/test_10/test_10_raw.mp4')

cap = cv2.VideoCapture('/dev/video2')
cap.set(cv2.CAP_PROP_FPS, fps)
cap.set(3, resolution[0])
cap.set(4, resolution[1])

out_video = cv2.VideoWriter(file_dir+file_name+'.mp4',cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)

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

try:
    
    outputfile = open(file_dir+file_name+'.csv', 'w')
    csvwriter = csv.writer(outputfile)
    #Write Header
    csvwriter.writerow(['Steering Angle','Gas Pedal Pos','Brake Pedal Pos', 'Speed'])
    print("Writing csv file output.csv. Press Ctrl-C to exit...\n")

    last_steer = 0
    last_gas = 0
    last_brake = 0
    last_speed = 0
    
    while True:
        
        ret, frame = cap.read()
        
        if not ret:
            print('No Frame Detected')
            raise KeyboardInterrupt
        
        frame = invert_frame(frame)

        can_recv = p.can_recv()
        
        steering_angle_codes = []
        gas_pedal_codes = []
        brake_pedal_codes = []
        speed_codes = []
        for address, _, dat, src  in can_recv:
            
            #CONVERT DATA TO AN ARRAY OF HEX MESSAGES
            dat_array = ["\\x%02x" % i for i in dat]
            dat_hex = "".join("\\x%02x" % i for i in dat)

            #EXTRACT MESSAGES
            
            if hex(address) == sas_pin:
                steering_angle_codes = dat_array[sas_slice]
            
            elif hex(address) == gas_pin:
                gas_pedal_codes = dat_array[gas_slice]

            elif hex(address) == brake_pin:
                brake_pedal_codes = dat_array[brake_slice]

            elif hex(address) == speed_pin:
                speed_codes = dat_array[speed_slice]

        #SPLIT INTO INDIVIDUAL HEX MESSAGES
        angle_hex = "".join(steering_angle_codes).split('\\x')
        gas_hex = "".join(gas_pedal_codes).split('\\x')
        brake_hex = "".join(brake_pedal_codes).split('\\x')
        speed_hex = "".join(speed_codes).split('\\x')
        
        #DECODING HEX MESSAGES
        #MUST BE ADJUSTED FOR DIFFERENT VEHICLES

        #STEERING ANGLE DECODE
        steer_angle = 0
        try:
            hex_1 = str(angle_hex[1])[1]
            hex_2 = str(angle_hex[2])[0]
            hex_3 = str(angle_hex[2])[1]
            
            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))
            bin_3 = "{0:04b}".format(int(hex_3, 16))
            
            bin_full = bin_1+bin_2+bin_3
            
            bin_value = int(bin_full[1:], 2)
            
            if int(bin_full[0]) == 1:
                bin_value -= 2047
            
            steer_angle = bin_value*1.5
            last_steer = steer_angle
        
        except Exception as ex:
            steer_angle = last_steer
            pass
            
        #GAS PEDAL DECODE
        gas_pedal_pos = 0
        try:
            hex_1 = str(gas_hex[1])[0]
            hex_2 = str(gas_hex[1])[1]

            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))

            bin_full = bin_1+bin_2

            bin_value = int(bin_full, 2)

            gas_pedal_pos = bin_value * 0.005
            last_gas = gas_pedal_pos

        except Exception as ex:
            gas_pedal_pos = last_gas
            pass
            
        #BRAKE PEDAL DECODE
        brake_pedal_pos = 0
        try:
            hex_1 = str(brake_hex[1])[1]
            hex_2 = str(brake_hex[2])[0]
            hex_3 = str(brake_hex[2])[1]

            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))
            bin_3 = "{0:04b}".format(int(hex_3, 16))

            bin_full = bin_1+bin_2+bin_3

            bin_value = int(bin_full[1:], 2)

            brake_pedal_pos = bin_value / 2047.0
            last_brake = brake_pedal_pos

        except Exception as ex:
            brake_pedal_pos = last_brake
            pass

        #SPEED DECODE
        speed = 0
        try:
            hex_1 = str(speed_hex[1])[0]
            hex_2 = str(speed_hex[1])[1]
            hex_3 = str(speed_hex[2])[0]
            hex_4 = str(speed_hex[2])[1]

            bin_1 = "{0:04b}".format(int(hex_1, 16))
            bin_2 = "{0:04b}".format(int(hex_2, 16))
            bin_3 = "{0:04b}".format(int(hex_3, 16))
            bin_4 = "{0:04b}".format(int(hex_4, 16))

            bin_full = bin_1+bin_2+bin_3+bin_4

            bin_value = int(bin_full, 2)

            speed = bin_value * 0.01
            last_speed = speed

        except Exception as ex:
            speed = last_speed
            pass


        #FORMAT DATA
        steer_angle = float('{0:.4f}'.format(steer_angle))
        gas_pedal_pos = float('{0:.4f}'.format(gas_pedal_pos))
        brake_pedal_pos = float('{0:.4f}'.format(brake_pedal_pos))
        speed = float('{0:.4f}'.format(speed))
        
        csvwriter.writerow([steer_angle, gas_pedal_pos, brake_pedal_pos, speed])
        
        out_video.write(frame)

        visualization(frame, steer_angle, gas_pedal_pos, brake_pedal_pos, speed)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt
                    
except KeyboardInterrupt:
    
    print('Exiting')
    outputfile.close()
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    sys.exit(0)