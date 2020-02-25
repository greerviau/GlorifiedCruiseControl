def invert_frame(self, frame):
    (H,W) = frame.shape[:2]
    center = (W/2, H/2)
    M = cv2.getRotationMatrix2D(center, 180, 1.0)
    frame = cv2.warpAffine(frame, M, (W,H))
    return frame