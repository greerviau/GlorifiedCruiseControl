import cv2
'''
for i in range(100):
    cap = cv2.VideoCapture(i)
    ret, frame = cap.read()

    if ret:
        print(str(i)+ ' ' + str(cap.get(5)))

'''
resolution = (640, 360)

cap = cv2.VideoCapture(2)
cap.set(3, resolution[0])
cap.set(4, resolution[1])
cap.set(5, 30)

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

out_video = cv2.VideoWriter('test_vid.mp4',cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    out_video.write(frame)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out_video.release()
cv2.destroyAllWindows()


