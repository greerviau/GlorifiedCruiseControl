from road_vision import VPS
import numpy as np 
import cv2
import math
import itertools

cap = cv2.VideoCapture('project_video.mp4')
#cap = cv2.VideoCapture(cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

vps = VPS(show_visuals=False, objects=False, cache_size=5)

wheel = cv2.imread('steering_wheel.png',0)
wheel = cv2.resize(wheel, (200,200))
truth_wheel = np.zeros((wheel.shape[0],wheel.shape[1],3))
truth_wheel[:,:,1] = wheel
pred_wheel = np.zeros((wheel.shape[0],wheel.shape[1],3))
pred_wheel[:,:,2] = wheel

def bar_graph(data, labels, ylim, size):

    lst_colors = list(map(list, itertools.product([0, 255], repeat=3)))

    colors = list(np.array(lst_colors)[1:len(data)+1])

    bar_buf = 50 // len(data)

    graph_frame = np.zeros((240,240,3))

    x = 60+bar_buf

    for d,l,c in zip(data,labels,colors):
        c = (int(c[0]), int(c[1]), int(c[2]))
        y = 200-int((d / ylim) * 200)
        w = int(bar_buf * 2)
        cv2.rectangle(graph_frame, (x,y), (x+w,200), c, -1)

        cv2.putText(graph_frame, l, (x, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

        x += w + bar_buf

    cv2.line(graph_frame, (40,0), (40,200),(255,255,255), 2)
    cv2.line(graph_frame, (40,200), (240,200), (255,255,255), 2)

    graph_frame = cv2.resize(graph_frame, size)

    return graph_frame

def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result 

angle = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = vps.road_vision(frame)

    visual_frame = np.zeros((700,1300,3), dtype=np.uint8)

    #VPS 
    visual_frame[:frame.shape[0],:frame.shape[1]] = frame
    #Wheel graphics
    truth_wheel_cpy = rotate(truth_wheel, angle)
    pred_wheel_cpy = rotate(pred_wheel, angle)
    visual_frame[100:100+truth_wheel_cpy.shape[0],700:700+truth_wheel_cpy.shape[1]] = truth_wheel_cpy
    visual_frame[100:100+pred_wheel_cpy.shape[0],700+truth_wheel_cpy.shape[1]+100:700+truth_wheel_cpy.shape[1]+100+pred_wheel_cpy.shape[1]] = pred_wheel_cpy

    angle += 1
    if angle > 360:
        angle = 0

    graph_1 = bar_graph([50,30,70,40,20,10], ['a','b','c','d','e','f'], 100, (300,300))

    visual_frame[400:400+graph_1.shape[0],50:50+graph_1.shape[1]] = graph_1

    #cv2.imshow('frame', frame)
    cv2.imshow('Interface', visual_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
