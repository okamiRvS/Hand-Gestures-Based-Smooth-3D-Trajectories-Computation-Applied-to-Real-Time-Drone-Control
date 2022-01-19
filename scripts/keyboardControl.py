from djitellopy import tello
from sqlalchemy import false
import keyPressModule as kp
import time
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

flag = True

def getKeyboardInput2(vels):
    global x, y, z, yaw, a, height, totTime, flag, isWebcam

    const = 117/10 # forward speed in cm/2     (15cm/s)
    interval = 0.25
    dInterval = const*interval

    lr, fb, ud, yv = 0, 0, 0, 0

    for i, vel in enumerate(vels):
        # if first index of vels
        if totTime < vel[2]:

            lr, ud, _, _ = vel

            lr_speed = (lr/15) * const * interval
            lr_interval = int(lr_speed)

            ud_speed = (ud/15) * const * interval
            ud_interval = int(ud_speed)

            break

    print(interval)

    if totTime > vels[-1][2] + 2 and flag and not isWebcam:
        me.land()
        flag = False

    time.sleep(interval)
    totTime += interval

    lr = int(lr)
    ud = int(ud)

    if lr !=0 and ud !=0:
        x += lr_interval
        z -= ud_interval
    
    print(lr, fb, ud, yv, x, y, z)
    # print(lr_interval, ud_interval)

    return [lr, fb, ud, yv, x, y, z]
    

def getKeyboardInput() -> list:
    """
    Get keyboard input to move the drone a fixed speed using a controller.
    Speed can be increased or decresed dinamically.

    Arguments:
        me: this permits to takeoff or land the drone
        img: save this img if getKey('z')
    """

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

    if kp.getKey("e"): me.takeoff(); time.sleep(3) # this allows the drone to takeoff
    if kp.getKey("q"): me.land() # this allows the drone to land

    if kp.getKey('z'):
        img = me.get_frame_read().frame
        img = cv2.flip(img, 1)
        cv2.imwrite(f'src/tello_screenshots/{time.time()}.jpg', img)
        time.sleep(0.3)
        return [lr, fb, ud, yv, x, y, z]

    time.sleep(interval)
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

    y = np.zeros_like(resTraj[1])

    xz = np.concatenate(([resTraj[0]], [resTraj[2]]), axis=0)
    xz = xz / np.mean(xz) - 1

    fig = plt.figure() 
    plt.subplot(1, 2, 1)
    plt.title('XY')
    plt.xlabel('x')
    plt.ylabel('Y')
    plt.ylim(-1, 1) 
    plt.xlim(-1, 1)
    plt.plot(xz[0], y)

    plt.subplot(1, 2, 2)
    plt.title('XZ')
    plt.xlabel('x')
    plt.ylabel('Z')
    plt.ylim(-1, 1) 
    plt.xlim(-1, 1)
    plt.plot(xz[0], xz[1])

    plt.show()

    # move coordinates of traj in range from -50cm to 50cm
    xz = xz * 100

    # delta space in cm
    dspace = np.diff(xz)

    # scale time from 0 to 10
    istTime = resTraj[6] / np.max(resTraj[6]) * 10
    #istTime = resTraj[6]

    # delta time in secs
    dtime = np.diff(istTime)
    dtime[0] = istTime[1]

    # cm / s
    vels = dspace / dtime

    # we add time as last coordinate
    vels = np.concatenate( (vels, [istTime[1:]], [dtime] )).T

    return vels


isWebcam = True
fullControll = fullControllModule.FullControll()

kp.init()

me = tello.Tello()

if not isWebcam:
    me.connect()
    print(me.get_battery())
    me.streamon() # to get the stream image

    me.takeoff(); 
    time.sleep(3)
    me.move_up(30)
    print("let's start")

# Reset values
fullControll.autoSet(isWebcam)

# Get data from hand
resTraj = fullControll.run(me)

if not isWebcam:
    me.streamoff()

velocities = normalizeData(resTraj)

while True:

    # Control with joystick
    vals = getKeyboardInput()
    if not (vals[0] == vals[1] == vals[2] == vals[3] == 0):
        me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
        time.sleep(0.05)

    # Control with detected trajectory
    vals = getKeyboardInput2(velocities)
    imgXY = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
    imgXZ = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
    
    if points[-1][0] != vals[4] or points[-1][1] != vals[5] or points[-1][2] != vals[6]:
        points.append((vals[4], vals[5], vals[6]))

    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    drawXYPoints(imgXY, points)
    drawXZPoints(imgXZ, points)
    cv2.imshow("imgXY",imgXY)
    cv2.imshow("imgXZ",imgXZ)
    cv2.waitKey(1)