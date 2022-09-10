import cv2
import time
from djitellopy import Tello

w,h = (360,240)

tello = Tello()
tello.connect()

tello.streamon()


for i in range(20):
	
	startTime = time.time()
	while (time.time()-startTime) <= 3:
		frame_read = tello.get_frame_read()
		frame = frame_read.frame
		frame = cv2.resize(frame,(w,h))
		cv2.imshow("image",frame)
		cv2.waitKey(1)

	frame_read = tello.get_frame_read()
	frame = frame_read.frame
	frame = cv2.resize(frame,(w,h))
	cv2.imwrite("picture" + str(i+1) + ".jpg", frame)
	print("index: " + str(i))

