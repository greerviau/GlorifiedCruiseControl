import numpy as np
import cv2

class Queue():
    def __init__(self, frame, length):
        self.queue = [frame for i in range(length)]

    def add(self, frame):
        temp_frame = frame
        for i in range(len(self.queue)):
            temp_frame = self.queue[i]
            self.queue[i] = frame
            frame = temp_frame

    def get(self, index):
        return self.queue[index-1]

    def get_queue(self):
        return self.queue

cap = cv2.VideoCapture('project_video.mp4')
#cap = cv2.VideoCapture(0)

queue = None

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640,360), interpolation = cv2.INTER_AREA)

    cv2.imshow('frame',frame)

    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    frame = cv2.Canny(frame,100,200)

    cv2.imshow('edge',frame)

    if queue is None:
        queue = Queue(frame, 30)
    else:
        queue.add(frame)
        temporal = np.dstack(((queue.get(20) - queue.get(30)), (queue.get(10) - queue.get(20)), (frame - queue.get(10))))
        cv2.imshow('temporal',temporal)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
