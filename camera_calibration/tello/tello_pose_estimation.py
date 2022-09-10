import numpy as np
import cv2 as cv
import glob
import math
from djitellopy import Tello
from threading import Thread
import time

areaRange = [20000,40000]
xRange = 20
yRange = 50


from time import sleep
import RPi.GPIO as GPIO

DIR = 20   # Direction GPIO Pin
STEP = 21  # Step GPIO Pin
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
SPR = 48   # Steps per Revolution (360 / 7.5)

GPIO.setmode(GPIO.BCM)
GPIO.setup(DIR, GPIO.OUT)
GPIO.setup(STEP, GPIO.OUT)
GPIO.output(DIR, CW)


tello = Tello()
tello.connect()
tello.streamon()
tello.takeoff()
print("took off")
time.sleep(0.5)
tello.move_up(30)
print("moved up")
print("start process")



# Load previously saved data
with np.load('B.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx','dist','rvecs','tvecs')]



def draw(img, corners, imgpts):
    #origin(center)
    start = tuple(corners[0].ravel())
    #end = tuple(corners[11].ravel())
    #origin = ((start[0]+end[0])/2,(start[1]+end[1])/2)
    
    
    #xyz axes
    zero = tuple(imgpts[0].ravel())
    one = tuple(imgpts[1].ravel())
    two = tuple(imgpts[2].ravel())
    
    #int casting
    #origin_int = (int(origin[0]),int(origin[1]))
    start_int = (int(start[0]),int(start[1]))
    zero_int = (int(zero[0]),int(zero[1]))
    one_int = (int(one[0]),int(one[1]))
    two_int = (int(two[0]),int(two[1]))
    
    #draw origin
    img = cv.circle(img, start_int, 2, (255,255,255), cv.FILLED)
    
    #draw all points
    for i in range(12):
        coor = tuple(corners[i].ravel())
        coor = (int(coor[0]),int(coor[1]))
        img = cv.circle(img, coor, 1, (255,255,255), cv.FILLED)
    
    #draw axes
    img = cv.line(img, start_int, zero_int, (255,0,0), 5)
    img = cv.line(img, start_int, one_int, (0,255,0), 5)
    img = cv.line(img, start_int, two_int, (0,0,255), 5)

    return img

#criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((4*3,3), np.float32)
objp[:,:2] = np.mgrid[0:4,0:3].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

def pose_estimation():
    #take 10 samples of chessboards from videostream and average it
    i = 0
    beta_sum = 0
    x_sum = 0
    y_sum = 0
    z_sum = 0
    while i <= 0:

        frame_read = tello.get_frame_read()
        img = frame_read.frame
        #img = cv.resize(img,(w,h))
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (4,3), None)
        if ret == True:
            i += 1
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            
            
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            rmtx,_ = cv.Rodrigues(rvecs)
            beta = math.asin(-rmtx[2][0])*180/math.pi
            beta_sum += beta
            x_sum += tvecs[0][0]
            y_sum += tvecs[1][0]
            z_sum += tvecs[2][0]
            #print("translation vector: " + str(tvecs))
            #print("beta from arcsin(degrees): " +str(math.asin(-rmtx[2][0])*180/math.pi) + " degrees")
            #print("rotation matrix: " + str(rmtx))
            
            start = tuple(corners[0].ravel())
            end = tuple(corners[11].ravel())
            center = (int((start[0]+end[0])/2),int((start[1]+end[1])/2))
            area = int(abs((end[0]-start[0])*(end[1]-start[1])))
            img = cv.circle(img,center,2,(255,0,0),cv.FILLED)
            
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img,corners2,imgpts)
            cv.imshow('img',img)
            cv.waitKey(1)
            cv.imwrite('chessboard' + str(i)+'.png', img)
        else:
            cv.imshow('img',img)
            cv.waitKey(1)

    #cv.destroyAllWindows()

    x = int(x_sum/1)
    y = int(y_sum/1)
    z = int(z_sum/1)
    beta = int(beta_sum/1)
    
    print("x,y,z,beta: " + str(x) + ", " +str(y) + ", " + str(z) + ", " + str(beta))
    poseDict = {}
    poseDict['x'] = x
    poseDict['y'] = y
    poseDict['z'] = z
    poseDict['beta'] = beta
    poseDict['center'] = center
    poseDict['area'] = area
    return poseDict


time.sleep(1)
frame_read = tello.get_frame_read()
startTime = time.time()

while time.time()-startTime <=1:
    
    img = frame_read.frame
    cv.imshow("img",img)
    cv.waitKey(1)

for i in range(2):
    
    
    poseDict = pose_estimation()
    beta = poseDict['beta']
    x = poseDict['x']
    y = poseDict['y']
    z = poseDict['z']
    print("beta,x,y,z: "+str(beta)+", "+str(x)+", "+str(y)+", "+str(z))
    if beta>0:
        tello.rotate_counter_clockwise(int(beta/2))
    if beta<0:
        tello.rotate_clockwise(int(-beta/2))
        
    print("rotated")
    time.sleep(1)
    
    tello.go_xyz_speed(0,int(-x/2),0,10)
    time.sleep(1)
    
    
    
poseDict = pose_estimation()
beta = poseDict['beta']
x = poseDict['x']
y = poseDict['y']
z = poseDict['z']
tello.go_xyz_speed(int(z/2)-20,0,0,10)
time.sleep(1)
    


tello.move_up(40)
time.sleep(0.5)

tello.move_forward(50)
time.sleep(0.5)
tello.go_xyz_speed(-5,0,0,10)
time.sleep(0.5)
tello.land()

print("drone landed")



step_count = SPR
delay = .0208

for x in range(step_count):
    GPIO.output(STEP, GPIO.HIGH)
    sleep(delay)
    GPIO.output(STEP, GPIO.LOW)
    sleep(delay)

sleep(.5)
GPIO.output(DIR, CCW)
for x in range(step_count):
    GPIO.output(STEP, GPIO.HIGH)
    sleep(delay)
    GPIO.output(STEP, GPIO.LOW)
    sleep(delay)

GPIO.cleanup()

print("process finished")

