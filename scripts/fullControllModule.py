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

import matplotlib.pyplot as plt

from screeninfo import get_monitors


class FullControll():

    def getKeyboardInput(self, me: tello.Tello, img: np.array) -> list:
        """
        Get keyboard input to move the drone a fixed speed using a controller.
        Speed can be increased or decresed dinamically.
        Arguments:
            me: this permits to takeoff or land the drone
            img: save this img if getKey('z')
        """

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


    def isWebcamOrDrone(self, me):
        """
        This function set parameters to work with webcam or drone camera
        """

        # HERE MAYBE COULD BE USEFUL USE A FACTORY FUNCTION (FROM SOFTWARE ENGENEERING)
        if self.getFromWebcam:
            
            # OPEN WEBCAM
            cv2.namedWindow(self.nameWindowWebcam)

            cv2.moveWindow(self.nameWindowWebcam, 0, int( get_monitors()[0].height / 2 ) + 10)

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
                if self.resize:
                    self.tracking.setSize(self.xResize, self.yResize)
                    self.normalizedPoints.setSize(self.xResize, self.yResize)
                else:
                    height, width, _ = img.shape
                    self.tracking.setSize(height, width)
                    self.normalizedPoints.setSize(height, width)
            else:
                success = False
            
            return img, cap

        else:
            # set size
            img = me.get_frame_read().frame
            if self.resize:
                self.tracking.setSize(self.xResize, self.yResize)
            else:
                height, width, _ = img.shape
                self.tracking.setSize(height, width)

            return img, None
            

    def closekp(self):
        kp.close()

    def run(self, me=None):
        """
        Execute the algorithm to detect the 3D trajectories from 2D hand landmarks
        """

        if not self.isSimulation:
            kp.init()

        # define variable to compute framerate
        pTime = 0
        cTime = 0

        img, cap = self.isWebcamOrDrone(me)

        # Save video from camera
        height, width = self.getResolution()
        video = cv2.VideoWriter(f'{self.path}_webcam.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        while True:

            if self.getFromWebcam:
                success, img = cap.read()
                img = cv2.flip(img, 1)

            else:

                img = me.get_frame_read().frame
                img = cv2.flip(img, 1)

                # print drone battery on screen
                fontScale = 1
                font = cv2.FONT_HERSHEY_DUPLEX
                thickness = 1
                color = (0,0,255)
                img = cv2.putText(img, 
                                    f"Battery: {me.get_battery()}", 
                                    (10, self.tracking.height-5),
                                    font, 
                                    fontScale, 
                                    color, 
                                    thickness)

            if self.resize:
                img = cv2.resize(img, (self.xResize, self.yResize)) # comment to get bigger frames
            
            # Control with joystick
            if not self.isSimulation:
                vals = self.getKeyboardInput(me, img)
                me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
                #print(f"vals are :{vals[0]}, {vals[1]}, {vals[2]}, {vals[3]}")
            
            img = self.detector.findHands(img, drawHand="LEFT")
            lmList = self.detector.findPosition(img, draw=False)

            if len(lmList) != 0:
                # setArray, computeMean, normalize points, and draw
                self.normalizedPoints.setArray(lmList)
                self.normalizedPoints.normalize()
                if self.allHandTransformed:
                    self.normalizedPoints.drawAllHandTransformed(img)
                self.normalizedPoints.removeHomogeneousCoordinate()

                # hand gesture recognition
                img, outputClass, probability = self.gestureDetector.processHands(img, self.normalizedPoints)
                res = self.tracking.run(img, self.normalizedPoints, outputClass, probability)
                
                if res is not None:
                    # Close video and return data
                    video.release()
                    return res
            else:
                self.tracking.justDrawLast2dTraj(img)

            # Update framerate
            cTime = time.time()
            fps = 1/(cTime-pTime)
            pTime = cTime

            fontScale = 1
            font = cv2.FONT_HERSHEY_DUPLEX
            thickness = 1
            cv2.putText(img, f"FPS: {int(fps)}", (10,40), font, fontScale, (255,0,255), thickness) # print fps

            # Write the flipped frame
            video.write(img)

            # Show frame
            cv2.imshow(self.nameWindowWebcam, img)
            key = cv2.waitKey(1)
            if key == 27: # exit on ESC
                break


    def getResolution(self):
        return self.tracking.height, self.tracking.width


    def autoSet(self, path, isWebcam=True, resize=False, showPlot=True, isSimulation=False, allHandTransformed=False):

        # Set if webcam or drone camera source
        # True is webcam, False is drone camera
        self.getFromWebcam = isWebcam

        # Set name window of imshow
        nameWindowWebcam = "Image"

        # Set if resize input img
        # if resize is True then width = xResize and height = yResize
        xResize = 360
        yResize = 240

        # Istantiate handDetector obj
        detector = htm.handDetector()

        #Istantiate handGestureRecognition obj
        gestureDetector = hgm.handGestureRecognition()

        # Istantiate normalizePoints obj
        normalizedPoints = normalize.normalizePoints()

        # Create a queue obj of a certain length 
        queue = qm.queueObj(lenMaxQueue=35)

        # Instantite tracking obj
        tracking = tm.tracking(queue, 
                                skipEveryNsec=0.25, #0
                                skipEveryNpoints=2, #4
                                trajTimeDuration=20, # trajTimeDuration is in seconds
                                log3D=showPlot) 

        # set variable
        self.nameWindowWebcam = nameWindowWebcam
        self.resize = resize
        self.xResize = xResize
        self.yResize = yResize
        self.detector = detector
        self.gestureDetector = gestureDetector
        self.normalizedPoints = normalizedPoints
        self.tracking = tracking
        self.isSimulation = isSimulation
        self.allHandTransformed = allHandTransformed
        self.path = path


def main():

    isWebcam = True
    me = tello.Tello()
    
    if not isWebcam:
        me.connect()
        print(me.get_battery())

    fullControll = FullControll()
    fullControll.autoSet(isWebcam=isWebcam, resize=True, showPlot=True)

    fullControll.run(me)

    if not isWebcam:
        me.streamoff()


if __name__ == "__main__":
    
    main()