import pandas as pd 
import numpy as np 

data = pd.read_csv('data/test_01/test_01.csv')

print(data.head())

messages = data['Message'].to_numpy()

messages = list(messages)

angle_message = []
for message in messages:
    angle_message.append(message.split('\\x')[1:3])

print(angle_message)

for ang in angle_message:
    steer_angle = int('0x'+str(ang[0])+str(ang[1]),0)
    
    #steer_angle =  bin
    print(steer_angle/360)
    