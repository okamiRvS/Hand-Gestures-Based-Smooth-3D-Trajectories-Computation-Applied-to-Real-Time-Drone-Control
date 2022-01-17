from djitellopy import tello
from sqlalchemy import false
import keyPressModule as kp
from time import sleep
import cv2
import numpy as np
import math
import pdb
import main as fullControllModule
import matplotlib.pyplot as plt


######################################################
# PARAMETERS
fSpeed = 117/10 # forward speed in cm/2     (15cm/s)
aSpeed = 360/10 # angular speed degrees/s   (50d/s)
interval = 0.25

dInterval = fSpeed*interval
aInterval = aSpeed*interval

######################################################
x, y, z = 500, 500, 500
a = 0
yaw = 0
height = 0
totTime = 0

points = [(500, 500, 500)]

# velocity (x,y,time)
vels = np.array([[90,90, 3], [-90,90, 6], [-90,-90, 9], [90,-90, 12]])
flag = True

kp.init()
me = tello.Tello()
# me.connect()
# print(me.get_battery())
#me.streamon() # to get the stream image

def getKeyboardInput2():
    global x, y, z, yaw, a, height, totTime, vels, flag

    fSpeed = 117/10 # forward speed in cm/2     (15cm/s)
    interval = 0.25
    dInterval = fSpeed*interval

    lr, fb, ud, yv = 0, 0, 0, 0

    for i, vel in enumerate(vels):
        # if first index of vels
        if totTime < vels[0][2]:
            lr, ud, _ = vels[0] * (1 / vels[0][2] )
            
            lr_interval_coef = lr/15
            lr_interval = int(dInterval * lr_interval_coef)
            ud_interval_coef = ud/15
            ud_interval = int(dInterval * ud_interval_coef)

        elif totTime < vel[2]:
            dtime = vel[2] - vels[i-1][2] 

            lr, ud, _ = vel * (1 / dtime )
            lr_interval_coef = lr/15
            lr_interval = int(dInterval * lr_interval_coef)
            ud_interval_coef = ud/15
            ud_interval = int(dInterval * ud_interval_coef)

            break

    if totTime > vels[-1][2] + 2 and flag:
        me.land()
        flag = False

    sleep(interval)
    totTime += interval

    lr = int(lr)
    ud = int(ud)

    if lr !=0 and ud !=0:
        x += lr_interval
        z -= ud_interval
    
    print(lr, fb, ud, yv, x, y, z)
    # print(lr_interval, ud_interval)

    return [lr, fb, ud, yv, x, y, z]
    

def getKeyboardInput():
    #left-right, foward-back, up-down, yaw veloity
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 15
    aSpeed = 50
    global x, y, z, yaw, a, height, totTime
    d = 0 # distance will be reset each time
    height = 0 #height will be reset each time

    if kp.getKey("LEFT"): 
        lr = -speed
        d = dInterval
        a = -180

    elif kp.getKey("RIGHT"): 
        lr = speed
        d = -dInterval
        a = 180

    if kp.getKey("UP"):  # this is forward, not up...
        fb = speed
        d = dInterval
        a = 270

    elif kp.getKey("DOWN"): # this is backward, not down...
        fb = -speed
        d = -dInterval
        a = -90

    if kp.getKey("w"): 
        ud = speed
        height = -dInterval

    elif kp.getKey("s"): 
        ud = -speed
        height = dInterval

    if kp.getKey("a"): 
        yv = -aSpeed
        yaw -= aInterval

    elif kp.getKey("d"): 
        yv = aSpeed
        yaw += aInterval

    if kp.getKey("e"): me.takeoff(); sleep(3) # this allows the drone to takeoff
    if kp.getKey("q"): me.land() # this allows the drone to land

    sleep(interval)
    totTime += interval

    a += yaw
    x += int(d*math.cos(math.radians(a)))
    y += int(d*math.sin(math.radians(a)))
    z += int(height)

    return [lr, fb, ud, yv, x, y, z]

def drawXYPoints(img, points):
    for point in points:
        cv2.circle(img, (point[0], point[1]), 5, (0,0,255), cv2.FILLED)
    
    # print last point in green
    cv2.circle(img, (points[-1][0], points[-1][1]), 8, (0,255,0), cv2.FILLED)

    # print coordinate of the last position
    cv2.putText(img,
                f"({ (points[-1][0] - 500) / 100}, { (points[-1][1] - 500) /100}, { (points[-1][2] - 500) /100})m",
                ( points[-1][0]+10, points[-1][1]+30 ),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,0,255),
                1) # this give us position in meters not in cm

    # print totTime
    cv2.putText(img, f"{totTime}s",
            ( 10, 30 ),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0,255,0),
            1) # this give us position in meters not in cm

def drawXZPoints(img, points):
    for point in points:
        cv2.circle(img, (point[0], point[2]), 5, (0,0,255), cv2.FILLED)
    
    # print last point in green
    cv2.circle(img, (points[-1][0], points[-1][2]), 8, (0,255,0), cv2.FILLED)

    # print coordinate of the last position
    cv2.putText(img,
                f"({ (points[-1][0] - 500) / 100}, { (points[-1][1] - 500) /100}, { (points[-1][2] - 500) /100})m",
                ( points[-1][0]+10, points[-1][2]+30 ),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255,0,255),
                1) # this give us position in meters not in cm

    # print totTime
    cv2.putText(img, f"{totTime}s",
            ( 10, 30 ),
            cv2.FONT_HERSHEY_PLAIN,
            3,
            (0,255,0),
            1) # this give us position in meters not in cm

def normalizeData(resTraj):

    y = resTraj[0] / np.mean(resTraj[0]) - 1
    x = resTraj[1] / np.mean(resTraj[1]) - 1
    z = resTraj[2] / np.mean(resTraj[2]) - 1
    
    fig = plt.figure() 
    plt.subplot(1, 2, 1)
    plt.title('XY')
    plt.xlabel('x')
    plt.ylabel('Y')
    plt.plot(x, y)

    plt.subplot(1, 2, 2)
    plt.title('XZ')
    plt.xlabel('x')
    plt.ylabel('Z')
    plt.plot(x, z)

    plt.show()
    pdb.set_trace()


# me.takeoff(); 
# sleep(3)
print("let's start")

fullControll = fullControllModule.FullControll()

while True:
    # Reset values
    fullControll.autoSet()

    # Get data from hand
    resTraj = fullControll.run()

    normalizeData(resTraj)

    vals = getKeyboardInput2()
    imgXY = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
    imgXZ = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
    
    if points[-1][0] != vals[4] or points[-1][1] != vals[5] or points[-1][2] != vals[6]:
        points.append((vals[4], vals[5], vals[6]))
    #print(points)

    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    drawXYPoints(imgXY, points)
    drawXZPoints(imgXZ, points)
    cv2.imshow("imgXY",imgXY)
    cv2.imshow("imgXZ",imgXZ)
    cv2.waitKey(1)