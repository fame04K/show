import cv2
import time

cap = cv2.VideoCapture(0)

for i in range(20):
    
    if cap.isOpened():
        startTime = time.time()
        while (time.time()-startTime) <= 3:
            
            ret, frame = cap.read()
            cv2.imshow("image",frame)
            cv2.waitKey(1)
    
        cv2.imwrite("webcam_picture" + str(i+1) + ".jpg", frame)
        print("index: " + str(i))
