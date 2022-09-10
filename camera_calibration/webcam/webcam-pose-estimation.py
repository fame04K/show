import numpy as np
import cv2 as cv
import glob
import math

w,h = (360,240)

cap = cv.VideoCapture(0)

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
    img = cv.circle(img, start_int, 10, (255,255,255), cv.FILLED)
    
    #draw all points
    for i in range(12):
        coor = tuple(corners[i].ravel())
        coor = (int(coor[0]),int(coor[1]))
        img = cv.circle(img, coor, 5, (255,255,255), cv.FILLED)
    
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


#take 10 samples of chessboards from videostream
i = 0
while i <= 0: #9
    if not cap.isOpened():
        continue
    _,img = cap.read()
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (4,3), None)
    if ret == True:
        i += 1
        corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        
        
        # Find the rotation and translation vectors.
        ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
        rmtx,_ = cv.Rodrigues(rvecs)
        print("beta from arcsin(degrees): " +str(math.asin(-rmtx[2][0])*180/math.pi) + " degrees")
        #print("rotation matrix: " + str(rmtx))
        
        
        # project 3D points to image plane
        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
        img = draw(img,corners2,imgpts)
        cv.imshow('img',img)
        cv.waitKey(500)
        cv.imwrite('chessboard' + str(i)+'.png', img)
    else:
        cv.imshow('img',img)
        cv.waitKey(1)
cv.destroyAllWindows()
