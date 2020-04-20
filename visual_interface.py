from cache import Cache
import numpy as np 
import cv2
import math
import time
import random as rnd
import pandas as pd
import pickle as pkl
import scipy
from utils import *
from sklearn.metrics import r2_score
from tensorflow.keras.models import load_model

video = cv2.VideoCapture('data_cleaned/sess_10/split_1/split_1.mp4')
obd_data = pd.read_csv('data_cleaned/sess_10/split_1/split_1.csv')
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
video.set(3, 1280)
video.set(4, 720)

from conv_net_model import *
network = conv_net(x, keep_prob)
saver = tf.train.Saver()

sess = tf.Session()
saver.restore(sess, 'conv_net/conv_net.ckpt')
'''

from road_vision import VPS
vps = VPS(objects=False, return_data=True)
dnn_model = load_model('dnn.h5')
'''
#pred_model = pkl.load(open('regression_model.pkl', 'rb'))

wheel = cv2.imread('steering_wheel.png',0)
wheel = cv2.resize(wheel, (200,200))
truth_wheel = np.zeros((wheel.shape[0],wheel.shape[1],3))
truth_wheel[:,:,1] = wheel
pred_wheel = np.zeros((wheel.shape[0],wheel.shape[1],3))
pred_wheel[:,:,2] = wheel

angle = 0

font = cv2.FONT_HERSHEY_SIMPLEX

#inter_out = cv2.VideoWriter("visual_interface.mp4",cv2.VideoWriter_fourcc(*'XVID'), 30,(1300,700))

graphs = 50
add_to_graph = 1

truth_cache = Cache(max_size=100)
pred_cache = Cache(max_size=100)
error_cache = Cache(max_size=100)
r2_cache = Cache(max_size=100)

frame_count = 0
prev_steering_angle_pred = 0

while True:
    ret, frame = video.read()
    if not ret:
        break
    
    steering_angle = obd_data.iloc[frame_count,1]
    frame_count+=1

    '''
    lane_data, vehicle_data, frame = vps.road_vision(frame)
    #print(steering_angle)

    #(mean_left_curve, mean_right_curve, mean_lane_curve, mean_vehicle_offset, mean_center_angle, median_left_curve, median_right_curve, median_lane_curve, median_vehicle_offset, median_center_angle, turn)

    lane_curve = lane_data[7] / 5000
    vehicle_offset = lane_data[8]
    center_angle = lane_data[9] * scipy.pi / 180
    X_input = [lane_curve, vehicle_offset, center_angle]
    #print(X_input)
    X_input = np.array([X_input])
    #print(X_input.shape)
    #print(X_input)
    
    steering_angle_pred = dnn_model.predict(X_input)[0][0] * 180 / scipy.pi

    steering_angle_pred = round(steering_angle_pred * 2) / 2
    '''

    input_frame = frame[frame.shape[0]//2:, :, :]
    input_frame = cv2.resize(input_frame, (200, 60)) / 127.5 - 1.0
    input_frame = input_frame.astype(np.float16)
    steering_angle_pred = sess.run(network, feed_dict={x:[input_frame], keep_prob:1.0})[0][0] * 180 / scipy.pi
    steering_angle_pred = round(steering_angle_pred * 2) / 2

    visual_frame = np.zeros((700,1300,3), dtype=np.uint8)
    #Video Frame
    visual_frame[:frame.shape[0],:frame.shape[1]] = frame
    #Wheel graphics
    cv2.putText(visual_frame, "Steering Angle", (865,50),font,0.7,(255,255,255),2)
    truth_wheel_cpy = rotate(truth_wheel, steering_angle)
    pred_wheel_cpy = rotate(pred_wheel, steering_angle_pred)
    visual_frame[100:100+pred_wheel_cpy.shape[0],700:700+pred_wheel_cpy.shape[1]] = pred_wheel_cpy
    visual_frame[100:100+truth_wheel_cpy.shape[0],700+pred_wheel_cpy.shape[1]+100:700+pred_wheel_cpy.shape[1]+100+truth_wheel_cpy.shape[1]] = truth_wheel_cpy
    
    textsize = cv2.getTextSize(str(steering_angle_pred), font, 0.7, 2)[0]

    offX = textsize[0] // 2
    offY = textsize[1] // 2

    cv2.putText(visual_frame, str(steering_angle_pred), (700+(truth_wheel_cpy.shape[1]//2)-offX, 100+(truth_wheel_cpy.shape[0]//2)+offY), font, 0.7, (0,0,0), 2)

    textsize = cv2.getTextSize(str(steering_angle), font, 0.7, 2)[0]

    offX = textsize[0] // 2
    offY = textsize[1] // 2

    cv2.putText(visual_frame, str(steering_angle), (700+truth_wheel_cpy.shape[1]+100+(pred_wheel_cpy.shape[1]//2)-offX, 100+(pred_wheel_cpy.shape[0]//2)+offY), font, 0.7, (0,0,0), 2)

    cv2.putText(visual_frame, 'Prediction', (740, 340), font, 0.7,(255,255,255),2)
    cv2.putText(visual_frame, 'Truth', (1070, 340), font, 0.7,(255,255,255),2)

    pred_cache.add([steering_angle_pred])
    truth_cache.add([steering_angle])

    ang_graph = angle_graph(truth_cache.get_all_index(0), pred_cache.get_all_index(0), (400,300), ['Angle', '-25', '25'], ['Time', '0', '100'])
    
    visual_frame[380:380+ang_graph.shape[0],30:30+ang_graph.shape[1]] = ang_graph

    error = abs(steering_angle - steering_angle_pred) / 10
    error_cache.add([error])

    acc_graph = line_graph(error_cache.get_all_index(0), error_cache.mean(0), 100, (400,300), ['Time', '0', '100'], ['Degrees of Error', '0', '10'])

    visual_frame[380:380+acc_graph.shape[0],430:430+acc_graph.shape[1]] = acc_graph

    r2 = r2_score(truth_cache.get_all_index(0), pred_cache.get_all_index(0)) / 100
    r2_cache.add([r2])
    #print(truth_cache.get_all_index(0))
    #print(pred_cache.get_all_index(0))
    #print(r2)

    r2_graph = r2_line_graph(r2_cache.get_all_index(0), r2_cache.mean(0), 100, (400,300), ['Time', '0', '100'], ['R2 Score', '-1', '1'])

    visual_frame[380:380+r2_graph.shape[0],850:850+r2_graph.shape[1]] = r2_graph

    #cv2.imshow('frame', frame)
    cv2.imshow('Interface', visual_frame)
    #inter_out.write(visual_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    time.sleep(0.02)

#inter_out.release()
video.release()
cv2.destroyAllWindows()
