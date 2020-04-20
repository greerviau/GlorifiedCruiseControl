import numpy as np 
import cv2 
import itertools
import math

def invert_frame(frame):
    (H,W) = frame.shape[:2]
    center = (W/2, H/2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    frame = cv2.warpAffine(frame, M, (W,H))
    return frame

def bar_graph(data, labels, ylim, size):

    font = cv2.FONT_HERSHEY_SIMPLEX

    lst_colors = list(map(list, itertools.product([0, 255], repeat=3)))

    colors = list(np.array(lst_colors)[1:len(data)+1])

    bar_buf = 50 // len(data)

    graph_frame = np.zeros((240,240,3))

    x = 60+bar_buf

    for d,l,c in zip(data,labels,colors):
        c = (int(c[0]), int(c[1]), int(c[2]))
        y = 200-int((float(d) / float(ylim)) * 200)
        w = int(bar_buf * 2)
        cv2.rectangle(graph_frame, (x,y), (x+w,200), c, -1)

        fontsize = 0.5

        while True:
            textsize = cv2.getTextSize(l, font, fontsize, 1)[0]
            offX = (w-textsize[0])//2
            if offX >= 0:
                break
            fontsize -= 0.05

        cv2.putText(graph_frame, l, (x+offX, 220), font, fontsize, (255,255,255), 1)

        x += w + bar_buf

    cv2.line(graph_frame, (40,0), (40,200),(255,255,255), 2)
    cv2.line(graph_frame, (40,200), (240,200), (255,255,255), 2)

    graph_frame = cv2.resize(graph_frame, size)

    return graph_frame

def line_graph(data, avg, xlim, size, x_label, y_label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    graph_frame = np.zeros((340,440,3))
    cv2.line(graph_frame, (40,300-int(avg*300)), (440,300-int(avg*300)), (0,0,255), 2)
    #print(len(data))
    spacing = math.floor(400 / xlim)
    current_point = data[0]
    for i, element in enumerate(data,1):
        next_point = element
        color = (0,255,0)
        #if next_point < avg:
         #   color = (0,0,255)
        cv2.line(graph_frame, (40+((i-1)*spacing), 300-int(current_point*300)), (40+(i*spacing), 300-int(next_point*300)), color, 2)
        current_point = next_point

    rot_text = np.zeros((40,200,3))
    textsize = cv2.getTextSize(str(y_label[0]), font, 0.7, 2)[0]
    cv2.putText(rot_text, y_label[0],(100-textsize[0]//2,25),font,0.7,(255,255,255),2)
    rot_text = np.rot90(rot_text,1)
    graph_frame[60:260,0:40] = rot_text

    textsize = cv2.getTextSize(str(x_label[0]), font, 0.7, 2)[0]
    cv2.putText(graph_frame,x_label[0],(40+(200-textsize[0]//2),325),font,0.7,(255,255,255),2)

    txt_w = cv2.getTextSize(str(y_label[1]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,y_label[1],(30-txt_w,300),font,0.4,(255,255,255),1)

    txt_w = cv2.getTextSize(str(y_label[2]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,y_label[2],(30-txt_w,10),font,0.4,(255,255,255),1)

    cv2.putText(graph_frame,x_label[1],(40,325),font,0.4,(255,255,255),1)

    txt_w = cv2.getTextSize(str(x_label[2]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,x_label[2],(440-txt_w,325),font,0.4,(255,255,255),1)

    cv2.line(graph_frame, (40,0), (40,300),(255,255,255), 2)
    cv2.line(graph_frame, (40,300), (440,300), (255,255,255), 2)

    graph_frame = cv2.resize(graph_frame, size)

    return graph_frame


def r2_line_graph(data, avg, xlim, size, x_label, y_label):
    font = cv2.FONT_HERSHEY_SIMPLEX
    graph_frame = np.zeros((340,440,3))
    cv2.line(graph_frame, (40,150-int(avg*300)), (440,150-int(avg*300)), (0,0,255), 2)
    #print(len(data))
    spacing = math.floor(400 / xlim)
    current_point = data[0]
    for i, element in enumerate(data,1):
        next_point = element
        color = (0,255,0)
        #if next_point < avg:
         #   color = (0,0,255)
        cv2.line(graph_frame, (40+((i-1)*spacing), 150-int(current_point*300)), (40+(i*spacing), 150-int(next_point*300)), color, 2)
        current_point = next_point

    
    rot_text = np.zeros((40,200,3))
    textsize = cv2.getTextSize(str(y_label[0]), font, 0.7, 2)[0]
    cv2.putText(rot_text, y_label[0],(100-textsize[0]//2,25),font,0.7,(255,255,255),2)
    rot_text = np.rot90(rot_text,1)
    graph_frame[60:260,0:40] = rot_text

    textsize = cv2.getTextSize(str(x_label[0]), font, 0.7, 2)[0]
    cv2.putText(graph_frame,x_label[0],(40+(200-textsize[0]//2),325),font,0.7,(255,255,255),2)

    txt_w = cv2.getTextSize(str(y_label[1]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,y_label[1],(30-txt_w,300),font,0.4,(255,255,255),1)

    txt_w = cv2.getTextSize(str(y_label[2]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,y_label[2],(30-txt_w,10),font,0.4,(255,255,255),1)

    cv2.putText(graph_frame,x_label[1],(40,325),font,0.4,(255,255,255),1)

    txt_w = cv2.getTextSize(str(x_label[2]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,x_label[2],(440-txt_w,325),font,0.4,(255,255,255),1)

    cv2.line(graph_frame, (40,0), (40,300),(255,255,255), 2)
    cv2.line(graph_frame, (40,300), (440,300), (255,255,255), 2)

    graph_frame = cv2.resize(graph_frame, size)

    return graph_frame

def angle_graph(data_1, data_2, size, x_label, y_label):

    font = cv2.FONT_HERSHEY_SIMPLEX
    graph_frame = np.zeros((340,440,3))

    cv2.line(graph_frame, (240,300), (240,0), (255,255,255), 2)
    spacing = math.floor(300 / 100)
    cp_1 = data_1[0]*8
    cp_2 = data_2[0]*8
    count = 1
    for e1, e2 in zip(data_1, data_2):
        np_1 = e1*8
        np_2 = e2*8
        c1 = (0,255,0)
        c2 = (0,0,255)
        #if next_point < avg:
         #   color = (0,0,255)
        cv2.line(graph_frame, (int(40+(200+cp_1)), int(300-((count-1)*spacing))), (int(40+(200+np_1)), int(300-((count)*spacing))), c1, 2)
        cv2.line(graph_frame, (int(40+(200+cp_2)), int(300-((count-1)*spacing))), (int(40+(200+np_2)), int(300-((count)*spacing))), c2, 2)
        cp_1 = np_1
        cp_2 = np_2
        count+=1

    rot_text = np.zeros((40,200,3))
    textsize = cv2.getTextSize(str(y_label[0]), font, 0.7, 2)[0]
    cv2.putText(rot_text, y_label[0],(100-textsize[0]//2,25),font,0.7,(255,255,255),2)
    rot_text = np.rot90(rot_text,1)
    graph_frame[60:260,0:40] = rot_text

    textsize = cv2.getTextSize(str(x_label[0]), font, 0.7, 2)[0]
    cv2.putText(graph_frame,x_label[0],(40+(200-textsize[0]//2),325),font,0.7,(255,255,255),2)

    txt_w = cv2.getTextSize(str(y_label[1]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,y_label[1],(30-txt_w,300),font,0.4,(255,255,255),1)

    txt_w = cv2.getTextSize(str(y_label[2]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,y_label[2],(30-txt_w,10),font,0.4,(255,255,255),1)

    cv2.putText(graph_frame,x_label[1],(40,325),font,0.4,(255,255,255),1)

    txt_w = cv2.getTextSize(str(x_label[2]), font, 0.4, 1)[0][0]
    cv2.putText(graph_frame,x_label[2],(440-txt_w,325),font,0.4,(255,255,255),1)

    cv2.line(graph_frame, (40,0), (40,300),(255,255,255), 2)
    cv2.line(graph_frame, (40,300), (440,300), (255,255,255), 2)

    graph_frame = cv2.resize(graph_frame, size)

    return graph_frame


def rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result 