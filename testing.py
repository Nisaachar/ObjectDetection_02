import cv2
import numpy as np
from matplotlib import pyplot as plt
import os



if not os.path.exists('pos'):
        os.makedirs('pos')


for filename in os.listdir('dataset'):
    if filename.endswith(".jpg") or filename.endswith(".py"):
        
        img = cv2.imread(filename)
        grayscaled = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

        kernel = np.ones((5,5),np.float32)/25
        dst = cv2.filter2D(img,-1,kernel)

        cv2.imshow('original',img)
        cv2.imshow('Adaptive threshold',th)
        cv2.imshow('smoothed', dst)
        pic_num = 1
        resized_image = cv2.resize(th, (100, 100))
        cv2.imwrite("pos/"+str(pic_num)+".jpg",resized_image)
         

cv2.waitKey(0)
cv2.destroyAllWindows()