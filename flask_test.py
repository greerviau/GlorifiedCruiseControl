from flask import Response
from flask import Flask
from flask import render_template
import threading
import cv2
import time

output_frame = None
lock = threading.Lock()

app = Flask(__name__)

vs = cv2.VideoCapture(0)
vs.set(cv2.CAP_PROP_FPS, 30)
vs.set(3, 640)
vs.set(4, 360)
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

def generate():
    global output_frame, lock

    while True:
        with lock:
            if output_frame is None:
                continue
            
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)

            if not flag:
                continue
        
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    t = threading.Thread(target=capture_video)
    t.daemon = True
    t.start()

    app.run(host='127.0.0.1', port=8080, debug=True, threaded=True, use_reloader=False)

vs.stop()