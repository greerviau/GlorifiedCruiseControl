import numpy as np 
import pandas as pd 
from road_vision import VPS
import sys
import csv
import os
import cv2

cols = ['mean_left_curve', 'mean_right_curve', 'mean_lane_curve', 'mean_vehicle_offset', 'mean_center_angle',
        'median_left_curve', 'median_right_curve', 'median_lane_curve', 'median_vehicle_offset', 'median_center_angle' , 'turn', 
        'v1_conf', 'v1_type', 'v1_size', 'v1_lane', 'v1_x', 'v1_y', 
        'v2_conf', 'v2_type', 'v2_size', 'v2_lane', 'v2_x', 'v2_y', 
        'v3_conf', 'v3_type', 'v3_size', 'v3_lane', 'v3_x', 'v3_y', 
        'v4_conf', 'v4_type', 'v4_size', 'v4_lane', 'v4_x', 'v4_y', 
        'v5_conf', 'v5_type', 'v5_size', 'v5_lane', 'v5_x', 'v5_y', 
        'steering_angle', 'speed']

print('COLS: ', len(cols))

processed_data_dir = 'data_processed'

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

data_dir = 'data_cleaned'

session_folders = os.listdir(data_dir)

for session in session_folders:
    split_dir = os.path.join(data_dir, session)
    splits = os.listdir(split_dir)

    session_data_dir = os.path.join(processed_data_dir, session+'_processed')

    if not os.path.exists(session_data_dir):
        os.makedirs(session_data_dir)

    aggregate_data = pd.DataFrame(columns=cols)
    #print(aggregate_data.head())

    for split in splits:
        road_video = cv2.VideoCapture(os.path.join(split_dir, split, split+'.mp4'))

        csv_data = pd.read_csv(os.path.join(split_dir, split, split+'.csv'))

        vps = VPS( show_data = True, objects=False, return_data=True)
        
        frame_count = 0
        
        rows = []

        skip = 0
        if split == 'split_1':
            skip = 100

        while True:
            ret, frame = road_video.read()

            if not ret:
                break

            if frame_count >= skip:
                lane_data, vehicle_data, frame = vps.road_vision(frame)

                #LANE DATA FORMAT (mean_left_curve, mean_right_curve, mean_lane_curve, mean_vehicle_offset, mean_center_angle, median_left_curve, median_right_curve, median_lane_curve, median_vehicle_offset, median_center_angle, turn)

                obd_data = csv_data.iloc[frame_count,:].to_list()

                sas_angle = obd_data[1]

                speed = obd_data[4]

                row = []

                row.extend(list(lane_data))

                for vehicle in vehicle_data:
                    row.extend(list(vehicle))

                row.append(sas_angle)
                row.append(speed)
                
                rows.append(row)

                cv2.imshow('VPS', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_count += 1

        split_data = pd.DataFrame(rows, columns=cols)

        aggregate_data = aggregate_data.append(split_data, ignore_index=True)
        print(aggregate_data.head())
        road_video.release()
        cv2.destroyAllWindows()
    
    aggregate_data.to_csv(os.path.join(session_data_dir, session+'.csv'))