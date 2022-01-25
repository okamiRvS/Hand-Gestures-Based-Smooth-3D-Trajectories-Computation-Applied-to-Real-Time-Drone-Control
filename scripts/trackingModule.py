import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT
import trajectory as traj
import smoothingModule as sm
import copy
import os

class tracking():

    def __init__(self, queueObj, skipEveryNsec, skipEveryNpoints, trajTimeDuration, log3D):

        self.queueObj = queueObj 
        self.currentState = "INIZIALIZATION"
        self.tolleranceSTART = -1
        self.tolleranceTRACKING = 500 # 100 before
        self.nlastMovements = 5
        self.scale = 0
        self.previous_mean_distance = -1

        if log3D:
            self.drawTraj = d3dT.dynamic3dDrawTrajectory()
        self.log3D = log3D

        self.smoothing = sm.smoothing(skipEveryNpoints)

        self.traj = traj.trajectory(skipEveryNsec, trajTimeDuration)
        self.trajCOMPLETE = []

        self.height = 0
        self.width = 0

        self.timeToCatchAnotherTraj = 1

        self.idxPoint = 0
        self.delayToExecuteTrajectory = 50

        self.previousTmpTime = 0
        self.trajFlag = True

        self.startingPoint = None

        # This var permits to execute always rc 0 0 0 0, but if
        # it is False for that fram will not be executed.
        # This is always reset True in run function
        self.flag = True


    def drawLog(self, img: np.array, color: tuple, checkTollerance: float, val: float):
        """
        Draw Log in img, with a specific color. Print the tollerance
        of the curren state and the val.
        """

        fontScale = 1 * (self.width /640)
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        cv2.putText(img, f"TOLLERANCE: {checkTollerance:.2f}", (10,70), font, fontScale, color, thickness)
        cv2.putText(img, f"POSITION: {val}", (10,100), font, fontScale, color, thickness)
        cv2.putText(img, f"DEPTH: {round(self.scale, 3)}", (10,130), font, fontScale, color, thickness)
        cv2.putText(img, f"STATUS: {self.currentState}", (10,160), font, fontScale, color, thickness)


    def draw2dTraj(self, img: np.array, xdata: list, zdata: list, flag: bool = False):
        """
        Draw the 2d trajectory in the window.
        """

        if flag:
            pdb.set_trace()

        numberOfPoints = len(xdata)

        for i in range(1, numberOfPoints):
            self.endingPoint = ( int(xdata[i] * self.width), int( (1-zdata[i]) * self.height) )
            cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
            cv2.circle(img, self.endingPoint, radius=0, color=(0,255,0), thickness=-1)
            
            if i != 1:
                cv2.line(img, self.startingPoint, self.endingPoint, (0,165,255), thickness=2)
            
            self.startingPoint = self.endingPoint


    def setSize(self, height: int, width: int):
        """
        Set height and width of the picture.
        """
        
        self.height = height
        self.width = width


    def justDrawLast2dTraj(self, img: np.array):
        """
        Draw the 2d trajectory in the window. This function is called
        when lmlList is empty.
        """

        if self.startingPoint is not None:
            xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.traj.getData()
            self.draw2dTraj(img, xdata, zdata)


    def distanceFromMeanPoint(self, lmList, val):

        # mean of all distances from mean point val and hand landmark in lmList to gain 3d scale factor
        # if it's greater respect of the previous istance than the hand is closer to the camera, otherwise is farther
        # the hand should be totally stopped
        sum_distances = 0
        for hand_land in lmList:
            sum_distances += math.sqrt( (val[0] - hand_land[1] )**2 + (val[1] - hand_land[2])**2 )

        return sum_distances / 21


    def addTrajectoryPointAndSpeed(self, lmList, val, roll, yaw, pitch):

        # mean of all distances from mean point val and hand landmark in lmList
        current_mean_dist = self.distanceFromMeanPoint(lmList, val)

        if current_mean_dist > self.previous_mean_distance:
            self.scale += 1 * abs(current_mean_dist - self.previous_mean_distance) # greater if the difference is high
        else:
            self.scale -= 1 * abs(current_mean_dist - self.previous_mean_distance) # same

        #print(self.scale)
        self.previous_mean_distance = current_mean_dist

        # collect data to draw the 3d trajectory
        # scale X,Z data from 0 to 1; about scale factor I consider 50 values, but maybe it requires some major details...
        self.traj.addPoint(x = val[0] / self.width,
                           y = self.scale / 50,
                           z = 1 - (val[1] / self.height),
                           roll = roll,
                           yaw = yaw,
                           pitch = pitch)

        # compute istant speed
        currentSpeed = self.traj.computeIstantSpeed() 
        self.traj.setSpeed(currentSpeed)
                
        # add time elapsed from the beginning
        self.traj.addTimeElapsed()


    def cleanTraj(self):

        # clean everything
        self.scale = 0
        self.previous_mean_distance = 0
        self.traj.reset()

        if self.log3D:
            self.drawTraj.clean()

        self.currentState = "START"


    def executeTrajectory(self, img, xdata, zdata):

        numberOfPoints = len(xdata)

        for i in range(1, numberOfPoints):
            self.endingPoint = ( int(xdata[i] * self.height), int( (1-zdata[i]) * self.width) )
            cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
            cv2.circle(img, self.endingPoint, radius=0, color=(0,255,0), thickness=-1)
            
            if i != 1:
                cv2.line(img, self.startingPoint, self.endingPoint, (255,255,0), thickness=1)
            
            self.startingPoint = self.endingPoint

        positionDrone = ( int(xdata[self.idxPoint] * self.height), int( (1-zdata[self.idxPoint]) * self.width) )
        cv2.circle(img, positionDrone, radius=10, color=(0,0,255), thickness=10)


    def run(self, img, normalizedPoints, outputClass, probability, me):

        val = normalizedPoints.mean.astype(int)
        cv2.circle(img, (val[0], val[1]), radius=3, color=(0,255,0), thickness=3)

        lmList = normalizedPoints.lmList

        roll, yaw, pitch = normalizedPoints.computeOrientation()
        normalizedPoints.computeDistanceWristMiddleFingerTip(pitch)
        # IF YOU WANT PRINT PITCH, ROLL, YAW
        normalizedPoints.drawOrientationVector(img, roll, yaw, pitch)

        self.flag = True

        if "INIZIALIZATION" == self.currentState:
            # fill all the queue before START state
            if self.queueObj.isFullQueue():
                self.currentState = "START" #exit from INIZIALIZATION mode

            self.drawLog(img, (0,0,255), 0, val)

        elif "START" == self.currentState:
            x_mean, y_mean = self.queueObj.mean()
            cv2.circle(img, (x_mean,y_mean), radius=3, color=(255,0,0), thickness=3)
            checkStart = math.sqrt( (x_mean - val[0] )**2 + (y_mean - val[1])**2 )

            # initialize self.previous_mean_distance to get information about
            # distance camera-hand. Set 
            if self.previous_mean_distance == -1:
                self.previous_mean_distance = self.distanceFromMeanPoint(lmList, val)
                self.tolleranceSTART = int(self.previous_mean_distance) + 20

            if checkStart < self.tolleranceSTART and self.queueObj.checkGesture("detect"):
                self.idxPoint = 0
                cv2.circle(img, (val[0], val[1]), radius=self.tolleranceSTART, color=(0,0,255), thickness=1) # draw the tollerance inside 
                cv2.circle(img, (val[0], val[1]), radius=self.tolleranceSTART*2, color=(0,255,0), thickness=1) # draw the tollerance outside 
                
                if self.traj.checkIsPossibleAddPoint():
                    self.addTrajectoryPointAndSpeed(lmList, val, roll, yaw, pitch)
 
            elif checkStart < self.tolleranceSTART and self.queueObj.checkGesture("ok"): # execute last trajectory
                
                if len(self.trajCOMPLETE) > 0: # if trajectory exists

                    if self.trajFlag:
                        
                        # destory 3d figure
                        if self.log3D:
                            self.drawTraj.destroy()

                        self.previousTmpTime = time.time()
                        self.trajFlag = False 

                    # draw 2d trajectory
                    xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed = self.smoothing.smoothCalculation()

                    self.draw2dTraj(img, xdata, zdata, False)

                    # after n seconds return in this way draw2dTraj() works
                    if (time.time() - self.previousTmpTime) > 2:
                        # return data to ros
                        return xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed

                else:

                    # there is no trajectory in queue...
                    self.cleanTraj()
                    self.currentState = "INIZIALIZATION"

            elif self.tolleranceSTART < checkStart < self.tolleranceSTART*2 and self.queueObj.checkGesture("detect"):
                self.traj.saveLastNValues(nPoints = 5) # take only the last 20 points
                xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.traj.getData()

                if len(xdata) > 1: # because otherwise could give index out of range
                    self.currentState = "TRACKING"
                    self.startingPoint = ( int( xdata[1] * self.height),
                                           int( (1- zdata[1]) * self.width) )

                self.addTrajectoryPointAndSpeed(lmList, val, roll, yaw, pitch)

                self.drawLog(img, (0,255,0), checkStart, val)

            elif self.queueObj.checkGesture("up"):
                me.send_rc_control(0, 0, 25, 0)
                self.flag = False
                time.sleep(0.05)

            elif self.queueObj.checkGesture("down"):
                me.send_rc_control(0, 0, -25, 0)
                self.flag = False
                time.sleep(0.05)
            
            elif self.queueObj.checkGesture("forward"):
                me.send_rc_control(0, 25, 0, 0)
                self.flag = False
                time.sleep(0.05)

            elif self.queueObj.checkGesture("backward"):
                me.send_rc_control(0, -25, 0, 0)
                self.flag = False
                time.sleep(0.05)

            elif self.queueObj.checkGesture("left"):
                me.send_rc_control(25, 0, 0, 0)
                self.flag = False
                time.sleep(0.05)

            elif self.queueObj.checkGesture("right"):
                me.send_rc_control(-25, 0, 0, 0)
                self.flag = False
                time.sleep(0.05)

            elif self.queueObj.checkGesture("stop"):
                me.send_rc_control(0, 0, 0, 0)
                self.flag = False
                time.sleep(0.05)

            elif self.queueObj.checkGesture("land"):
                me.land()
                self.flag = False
                time.sleep(0.05)

            else:
                self.drawLog(img, (0,0,255), checkStart, val)

        elif "TRACKING" == self.currentState:
            # il vantaggio di mediare sugli ultimi valori è perché in questo modo se viene perso il flusso lo si può riprendere molto velocemente
            # se si riparte dallo stesso punto in cui si è perso il tracking, diversamente bisognerà riacquisire tante nuove posizioni finchè 
            # la nuova media non sia sotto la tolleranza

            x_mean, y_mean = self.queueObj.meanOfTheLastNelements(self.nlastMovements)
            cv2.circle(img, (x_mean,y_mean), radius=3, color=(255,0,0), thickness=3)
            checkStartTracking = math.sqrt( (x_mean - val[0] )**2 + (y_mean - val[1])**2 )

            if self.queueObj.checkGesture("ok"):
                self.currentState = "EXIT"

                # remove last n keypoint because the movement is thumbup, so the end of traj
                self.traj.thumbsUpFix(numberKeyPoints=2)

                currentTraj = copy.deepcopy(self.traj)
                self.trajCOMPLETE.append(currentTraj)

                # wait at least n seconds to catch another trajectory
                self.waitForNewTraj = time.time() + self.timeToCatchAnotherTraj

                # smooth every data
                xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.trajCOMPLETE[-1].getData()
                self.smoothing.setPoints(xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed)
                self.smoothing.skipEveryNpointsFunc()

                # this is useful otherwise there is overlap of old and new points
                if self.log3D:
                    self.drawTraj.clean()

            elif checkStartTracking < self.tolleranceTRACKING and self.queueObj.checkGesture("detect"):

                # if totaltime is under the time of trajectory execution
                # and if passed n secs to add a new point
                if self.traj.checkIsPossibleAddPoint() and self.traj.checkTrajTimeDuration():
                    self.addTrajectoryPointAndSpeed(lmList, val, roll, yaw, pitch)
            
            elif self.queueObj.checkGesture("stop"):
                # If stop gesture then reset
                self.cleanTraj()
                self.currentState = "INIZIALIZATION"
                
            xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.traj.getData()
            self.draw2dTraj(img, xdata, zdata)

            if self.log3D:
                self.drawTraj.run(xdata, ydata, zdata, directionx, directiony, directionz, speed)

            self.drawLog(img, (255,0,0), checkStartTracking, val)


        elif "EXIT" == self.currentState:

            #xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.trajCOMPLETE[-1].getData()
            xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed = self.smoothing.smoothCalculation()
            
            # if passed a few time from TRAKING state
            if self.waitForNewTraj < time.time():

                # If stop gesture then reset
                if self.queueObj.checkGesture("detect") or self.queueObj.checkGesture("stop"):
                    self.cleanTraj()
                    self.currentState = "INIZIALIZATION"

                # If thumbsup then execute action
                elif self.queueObj.checkGesture("ok"):
                    return xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed
                        
            self.draw2dTraj(img, xdata, zdata)

            if self.log3D:
                self.drawTraj.run(xdata, ydata, zdata, rolldata, yawdata, pitchdata, speed)

            self.drawLog(img, (0,0,255), 0, val)

        self.queueObj.addMeanAndMatch(val, outputClass, probability)

        # Send this to the drone otherwise it will land automatically
        if self.flag:
            me.send_rc_control(0, 0, 0, 0)



def main():

   print("hello")


if __name__ == "__main__":
    
    main()