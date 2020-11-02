import numpy as np
import cv2
import os
from tqdm import tqdm

data_dir = '/home/greer/Documents/GCC_Data/4Runner'

sessions = os.listdir(data_dir)

for session in sessions:
    print(session)
    images_dir = os.path.join(data_dir, session, 'images')
    images = os.listdir(images_dir)
    images.sort()
    t = tqdm(images)
    for img in t:
        img_dir = os.path.join(images_dir, img)
        image = cv2.imread(img_dir)
        image = cv2.resize(image, (640, 360))
        cv2.imwrite(img_dir, image)

        t.set_description_str(img)
        t.refresh()

