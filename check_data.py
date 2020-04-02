import os
import pandas as pd
import cv2

data_sum = 0
for folder in os.listdir('data/'):
    data = pd.read_csv('data/'+folder+'/'+folder+'.csv')
    data_sum += len(data)

print('Samples Collected: ', data_sum)

cap = cv2.VideoCapture('data/sess_01/sess_01.mp4')

while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow('frame', frame)
    cv2.waitKey(1)
