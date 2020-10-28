import numpy as np 
import pandas as pd 
import sys
import csv
import os
import cv2
import pickle
import scipy
import random

def normalize(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

processed_data_dir = 'data_processed_raw'

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

data_dir = 'data_cleaned'

session_folders = os.listdir(data_dir)

for session in session_folders:
    print()
    print(session)
    split_dir = os.path.join(data_dir, session)
    splits = os.listdir(split_dir)

    session_data_dir = os.path.join(processed_data_dir, session+'_processed')

    if not os.path.exists(session_data_dir):
        os.makedirs(session_data_dir)

    #print(aggregate_data.head())
    total_frames = []
    aggregate_data = []
    split = 0
    for split in splits:
        print()
        print(split)
        road_video = cv2.VideoCapture(os.path.join(split_dir, split, split+'.mp4'))

        wheel_data = pd.read_csv(os.path.join(split_dir, split, split+'.csv'))['Steering Angle'].to_numpy()  

        frame_count = 0
        pro_data = []
        print('\nStarting Size: ',len(wheel_data))

        skip = 0
        if split == 'split_1':
            skip = 100

        while True:
            ret, frame = road_video.read()

            if not ret:
                break
            
            if frame_count >= skip:
                size = frame.shape
                frame = frame[size[0]//2:, :, :]
                frame = cv2.resize(frame, (200, 60))
                
                #Normalize frame
                normal_frame = frame / 127.5 - 1.0

                pro_data.append([np.copy(normal_frame).astype(np.float16), wheel_data[frame_count]])

                cv2.imshow('Frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            frame_count += 1

        '''
        BALLANCE DATA
        To ballance data, split into left right and straigt samples
        Make all sample collections the same length
        '''
        left_samples = []
        right_samples = []
        center_samples = []
        for sample in pro_data:
            if sample[1] < 0:
                left_samples.append(sample)
            elif sample[1] > 0:
                right_samples.append(sample)
            else:
                center_samples.append(sample)
        print('Before Balancing')
        print('Left: ',len(left_samples))
        print('Right: ',len(right_samples))
        print('Straight: ',len(center_samples))

        limit = min(len(left_samples), len(right_samples), len(center_samples))

        left_samples = left_samples[:limit]
        right_samples = right_samples[:limit]
        center_samples = center_samples[:limit]

        left_samples = np.array(left_samples)
        right_samples = np.array(right_samples)
        center_samples = np.array(center_samples)

        frames = []
        pro_wheel_data = []
        if len(left_samples) > 0:
            pro_wheel_data.extend(list(left_samples[:,1]))
            frames.extend(list(left_samples[:,0]))
        
        if len(right_samples) > 0:
            pro_wheel_data.extend(list(right_samples[:,1]))
            frames.extend(list(right_samples[:,0]))
        
        if len(center_samples) > 0:
            pro_wheel_data.extend(list(center_samples[:,1]))
            frames.extend(list(center_samples[:,0]))
        
        pro_wheel_data = list(np.array(pro_wheel_data) * scipy.pi / 180)
        aggregate_data.extend(pro_wheel_data)
        total_frames.extend(frames)
        print('Ending Size: ',len(pro_wheel_data))
        print('After Balancing')
        print('Left: ',len(left_samples))
        print('Right: ',len(right_samples))
        print('Straight: ',len(center_samples))

        road_video.release()
        cv2.destroyAllWindows()

    #SHUFFLE DATA
    np.random.seed(547)
    np.random.shuffle(aggregate_data)
    np.random.seed(547)
    np.random.shuffle(total_frames)
    print('Num frames: {} Data length: {}'.format(len(total_frames), len(aggregate_data)))

    #SAVE AGGREGATE
    aggregate_data = pd.DataFrame(aggregate_data, columns=['Steering Angle'])
    aggregate_data.to_csv(os.path.join(session_data_dir, 'Y.csv'))
    total_frames = np.array(total_frames)
    np.save(os.path.join(session_data_dir, 'X'), total_frames)
