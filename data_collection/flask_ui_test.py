from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2
import time

output_frame = None
steering_angle = 50
lock = threading.Lock()

app = Flask(__name__)

vs = cv2.VideoCapture('test_1.mp4')
#vs.set(cv2.CAP_PROP_FPS, 30)
#vs.set(3, 640)
#vs.set(4, 360)
time.sleep(2.0)

@app.route('/')
def index():
    return render_template('index.html')

def capture_video():
    global vs, output_frame, lock
    while True:
        ret, frame = vs.read()
        frame = cv2.resize(frame, (640,360))
        with lock:
            output_frame = frame.copy()

def get_frame():
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                continue
            
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

def get_steering_angle():
    global steering_angle, lock

    while True:
        yield(steering_angle)

@app.route('/video_feed')
def video_feed():
    return Response(get_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/steering_angle')
def steering_angle():
    return Response(get_steering_angle())

if __name__ == "__main__":
    t = threading.Thread(target=capture_video)
    t.daemon = True
    t.start()

    app.run(host='127.0.0.1', port=8080, debug=True, threaded=True, use_reloader=False)

vs.stop()