from road_vision import VPS
from cache import Cache
import numpy as np 
import cv2
import math
import random as rnd
from utils import *



cap = cv2.VideoCapture('tests/test_10/test_10_raw.mp4')
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

vps = VPS(show_visuals=False, show_data = False, objects=True, return_data=True, readout=False, cache_size=5)

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

acc_cache = Cache(max_size=100)
acc = rnd.random()
acc_cache.add([acc])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    lane_data, vehicle_data, frame = vps.road_vision(frame)

    visual_frame = np.zeros((700,1300,3), dtype=np.uint8)

    #VPS 
    visual_frame[:frame.shape[0],:frame.shape[1]] = frame
    #Wheel graphics
    cv2.putText(visual_frame, "Steering Angle", (865,50),font,0.7,(255,255,255),2)
    truth_wheel_cpy = rotate(truth_wheel, angle)
    pred_wheel_cpy = rotate(pred_wheel, angle)
    visual_frame[100:100+pred_wheel_cpy.shape[0],700:700+pred_wheel_cpy.shape[1]] = pred_wheel_cpy
    visual_frame[100:100+truth_wheel_cpy.shape[0],700+pred_wheel_cpy.shape[1]+100:700+pred_wheel_cpy.shape[1]+100+truth_wheel_cpy.shape[1]] = truth_wheel_cpy

    textsize = cv2.getTextSize(str(angle), font, 0.7, 2)[0]

    offX = textsize[0] // 2
    offY = textsize[1] // 2

    cv2.putText(visual_frame, str(angle), (700+(truth_wheel_cpy.shape[1]//2)-offX, 100+(truth_wheel_cpy.shape[0]//2)+offY), font, 0.7, (0,0,0), 2)

    textsize = cv2.getTextSize(str(angle), font, 0.7, 2)[0]

    offX = textsize[0] // 2
    offY = textsize[1] // 2

    cv2.putText(visual_frame, str(angle), (700+truth_wheel_cpy.shape[1]+100+(pred_wheel_cpy.shape[1]//2)-offX, 100+(pred_wheel_cpy.shape[0]//2)+offY), font, 0.7, (0,0,0), 2)

    angle += 1
    if angle > 360:
        angle = 0

    d1 = [graphs, graphs] 
    d2 = [graphs, graphs] 

    if graphs == 100:
        add_to_graph = -1
    elif graphs == 1:
        add_to_graph = 1

    graphs += add_to_graph

    graph_1 = bar_graph(d1, ['Pred','Truth'], 100, (275,275))
    graph_2 = bar_graph(d2, ['Pred','Truth'], 100, (275,275))

    textsize = cv2.getTextSize("Throttle", font, 0.7, 2)[0]

    offX = textsize[0] // 2

    cv2.putText(visual_frame, "Throttle", (50+(graph_1.shape[1]//2),410),font,0.7,(255,255,255), 2)

    visual_frame[425:425+graph_1.shape[0],50:50+graph_1.shape[1]] = graph_1

    cv2.putText(visual_frame, "Brake", (50+graph_1.shape[1]+50+(graph_2.shape[1]//2),410),font,0.7,(255,255,255), 2)

    visual_frame[425:425+graph_2.shape[0],50+graph_1.shape[1]+50:50+graph_1.shape[1]+50+graph_2.shape[1]] = graph_2

    acc = rnd.random()
    #print([acc])
    acc_cache.add([acc])

    acc_graph = line_graph(acc_cache.get_all(), acc_cache.mean(0), 100, (400,300))

    visual_frame[380:380+acc_graph.shape[0],750:750+acc_graph.shape[1]] = acc_graph

    #cv2.imshow('frame', frame)
    cv2.imshow('Interface', visual_frame)
    #inter_out.write(visual_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#inter_out.release()
cap.release()
cv2.destroyAllWindows()
