import numpy as np
import pandas as pd
import cv2
import os
import sys
import time

def clipper(d_dir):
    path = os.path.join(data_dir,d_dir,d_dir)
    print(path+'.mp4')
    
    video = cv2.VideoCapture(path+'.mp4')
    
    split_at = [0]
    c = 0
    while True:

        ret, frame = video.read()

        if not ret:
            break

        c += 1
        
        cv2.imshow('Clipper', frame)
        time.sleep(0.01)

        if c % 1 == 0:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Key Frame: ',c)
                split_at.append(c)

    split_at.append(c)
    video.release()
    cv2.destroyAllWindows()
    return (d_dir, split_at)

def process_clips(clip_data):
    d_dir = clip_data[0]
    split_at = clip_data[1]

    path = os.path.join(data_dir,d_dir,d_dir)
    print(path+'.mp4')

    processed_path = os.path.join(cleaned_data_dir, d_dir)

    video = cv2.VideoCapture(path+'.mp4')
    data = pd.read_csv(path+'.csv')
    c = 0

    for split in range(1,len(split_at)):
        split_dir = os.path.join(processed_path, 'split_'+str(split))
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)
        out_video = cv2.VideoWriter(split_dir+'/split_'+str(split)+'.mp4',cv2.VideoWriter_fourcc(*'XVID'), 30, resolution)
        while True:
            ret, frame = video.read()
            if not ret:
                break

            out_video.write(frame)

            c+= 1

            if c == split_at[split]:
                out_video.release()
                print(split_at[split-1],split_at[split])
                data.iloc[split_at[split-1]:split_at[split],:].to_csv(split_dir+'/split_'+str(split)+'.csv')
                break

    video.release()

if __name__ == '__main__':

    data_dir = 'data'

    resolution = (640, 360)

    cleaned_data_dir = 'data_cleaned'

    if not os.path.exists(cleaned_data_dir):
        os.makedirs(cleaned_data_dir)

    data_folders = os.listdir(data_dir)

    if(len(sys.argv) > 1):
        split_info = clipper(sys.argv[1])
        process_clips(split_info)
    else:
        split_info = []
        for i, folder in enumerate(data_folders, 0):
            print('Clip {}/{}'.format(i+1, len(data_folders)))
            split_info.append(clipper(folder))

        for split_i in split_info:
            process_clips((split_i))
        
            
    
    
        