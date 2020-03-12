import numpy as np
import pandas as pd
import cv2
import os
import sys

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

        if c % 1 == 0:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Key Frame: ',c)
                split_at.append(c)

    split_at.append(c)
    video.release()

    processed_path = os.path.join(processed_data_dir, d_dir)

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

data_dir = 'data'

resolution = (640, 360)

processed_data_dir = 'data_processed'

if not os.path.exists(processed_data_dir):
    os.makedirs(processed_data_dir)

data_folders = os.listdir(data_dir)

if __name__ == '__main__':
    clipper(sys.argv[1])
else:
    for folder in data_folders:
        clipper(folder)
    
            
    
    
        