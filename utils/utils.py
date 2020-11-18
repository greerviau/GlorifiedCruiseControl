import cv2

def gstreamer_pipeline(capture_width=1920,capture_height=1080,display_width=1280,display_height=720,framerate=30,flip_method=0):
    return ("nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width={}, height={}, "
        "format=NV12, framerate={}/1 ! "
        "nvvidconv flip-method={} ! "
        "video/x-raw, width={}, height={}, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! appsink".format(
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height))