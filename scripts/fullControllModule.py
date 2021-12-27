from djitellopy import tello
import keyPressModule as kp
import time
import cv2
import handTrackingModule as htm
import handGestureModule as hgm
import queueModule as qm
import trackingModule as tm
import normalizePointsModule as normalize
from numba import jit
import numpy as np
import pdb
import os

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt

from screeninfo import get_monitors


class FullControll():

    def getKeyboardInput(self, me):

        #left-right, foward-back, up-down, yaw velocity
        lr, fb, ud, yv = 0, 0, 0, 0
        speed = 30

        if kp.getKey("LEFT"): lr = -speed
        elif kp.getKey("RIGHT"): lr = speed

        if kp.getKey("UP"): fb = speed
        elif kp.getKey("DOWN"): fb = -speed

        if kp.getKey("w"): ud = speed
        elif kp.getKey("s"): ud = -speed

        if kp.getKey("a"): yv = speed
        elif kp.getKey("d"): yv = -speed

        if kp.getKey("e"): me.takeoff(); time.sleep(3) # this allows the drone to takeoff
        if kp.getKey("q"): me.land() # this allows the drone to land

        if kp.getKey('z'):
            cv2.imwrite(f'src/tello_screenshots/{time.time()}.jpg', img)
            time.sleep(0.3)

        return [lr, fb, ud, yv]


    def main(self):

        global img

        detector = htm.handDetector()
        gestureDetector = hgm.handGestureRecognition()
        normalizedPoints = normalize.normalizePoints()

        queue = qm.queueObj(lenMaxQueue=35)
        tracking = tm.tracking(queue, skipEveryNpoints=4, trajTimeDuration=10, log3D=False) # trajTimeDuration is in seconds (40)

        pTime = 0
        cTime = 0

        resize = False
        xResize = 360
        yResize = 240
        
        getFromWebcam = True

        nameWindowWebcam = "Image"

        # HERE MAYBE COULD BE USEFUL USE A FACTORY FUNCTION (FROM SOFTWARE ENGENEERING)
        if getFromWebcam:
            
            # OPEN WEBCAM
            cv2.namedWindow(nameWindowWebcam)

            cv2.moveWindow(nameWindowWebcam, 0, int( get_monitors()[0].height / 2 ) + 10)

            # For Linux, make sure OpenCV is built using the WITH_V4L (with video for linux).
            # sudo apt install v4l-utils
            # https://www.youtube.com/watch?v=ec4-1gF-cNU
            if os.name == 'posix': # if linux system
                cap = cv2.VideoCapture(0)
            elif os.name == 'nt': # if windows system
                cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

            #Check if camera was opened correctly
            if cap.isOpened():

                # try to get the first frame
                success, img = cap.read()

                # set size
                if resize:
                    tracking.setSize(xResize, yResize)
                    normalizedPoints.setSize(xResize, yResize)
                else:
                    height, width, _ = img.shape
                    tracking.setSize(height, width)
                    normalizedPoints.setSize(height, width)
            else:
                success = False
        else:
            kp.init()
            me = tello.Tello()
            me.connect()
            print(me.get_battery())
            me.streamon() # to get the stream image

            # set size
            img = me.get_frame_read().frame
            if resize:
                tracking.setSize(xResize, yResize)
            else:
                height, width, _ = img.shape
                tracking.setSize(height, width)

        while True:

            if getFromWebcam:
                success, img = cap.read()
            else:
                vals = self.getKeyboardInput(me)
                if not (vals[0] == vals[1] == vals[2] == vals[3] == 0):
                    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
                    time.sleep(0.05)

                img = me.get_frame_read().frame

            if resize:
                img = cv2.resize(img, (xResize, yResize)) # comment to get bigger frames

            img = cv2.flip(img, 1)
            
            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # setArray, computeMean, normalize points, and draw
                normalizedPoints.setArray(lmList)
                normalizedPoints.normalize()
                normalizedPoints.drawAllHandTransformed(img)
                normalizedPoints.removeHomogeneousCoordinate()

                # hand gesture recognition
                img, outputClass, probability = gestureDetector.processHands(img, normalizedPoints)
                res = tracking.run(img, normalizedPoints, outputClass, probability)
                
                if res is not None:
                    return res
            else:
                tracking.justDrawLast2dTraj(img)


            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            fontScale = 1
            font = cv2.FONT_HERSHEY_DUPLEX
            thickness = 1
            cv2.putText(img, f"FPS: {int(fps)}", (10,40), font, fontScale, (255,0,255), thickness) # print fps

            cv2.imshow(nameWindowWebcam, img)
            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break


if __name__ == "__main__":
    
    fullControll = FullControll()
    fullControll.main()