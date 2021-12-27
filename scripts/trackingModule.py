import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT
import trajectory as traj
import smoothingModule as sm
import copy
import pandas as pd
import os

class tracking():

    def __init__(self, queueObj, skipEveryNpoints, trajTimeDuration, log3D):

        self.queueObj = queueObj 
        self.currentState = "INIZIALIZATION"
        self.tolleranceSTART = 80
        self.tolleranceTRACKING = 500 # 100 before
        self.nlastMovements = 5
        self.scale = 0
        self.previous_mean_distance = 0

        if log3D:
            self.drawTraj = d3dT.dynamic3dDrawTrajectory()
        self.log3D = log3D

        self.smoothing = sm.smoothing()

        self.traj = traj.trajectory(skipEveryNpoints, trajTimeDuration)
        self.trajCOMPLETE = []

        self.height = 0
        self.width = 0

        self.timeToCatchAnotherTraj = 5

        self.idxPoint = 0
        self.delayToExecuteTrajectory = 50

        self.previousTmpTime = 0
        self.trajFlag = True


    def drawLog(self, img, color, checkTollerance, val):

        fontScale = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        cv2.putText(img, f"checkTollerance: {checkTollerance:.2f}", (10,70), font, fontScale, color, thickness)
        cv2.putText(img, f"val: {val}", (10,100), font, fontScale, color, thickness)
        cv2.putText(img, f"scale: {round(self.scale, 3)}", (10,130), font, fontScale, color, thickness)
        cv2.putText(img, f"{self.currentState}", (10,160), font, fontScale, color, thickness)


    def draw2dTraj(self, img, xdata, zdata, flag = False):

        if flag:
            pdb.set_trace()

        numberOfPoints = len(xdata)

        for i in range(1, numberOfPoints):
            self.endingPoint = ( int(xdata[i] * self.width), int( (1-zdata[i]) * self.height) )
            cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
            cv2.circle(img, self.endingPoint, radius=0, color=(0,255,0), thickness=-1)
            
            if i != 1:
                cv2.line(img, self.startingPoint, self.endingPoint, (255,255,0), thickness=1)
            
            self.startingPoint = self.endingPoint


    def setSize(self, height, width):
        
        self.height = height
        self.width = width


    def justDrawLast2dTraj(self, img):

        xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.traj.skipEveryNpointsFunc()
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

        self.traj.addPoint(x = val[0] / self.width,
                           y = self.scale / 50,
                           z = 1 - (val[1] / self.height),
                           roll = roll,
                           yaw = yaw,
                           pitch = pitch)
        currentSpeed = self.traj.computeIstantSpeed() # compute istant speed
        self.traj.setSpeed(currentSpeed)


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


    def run(self, img, normalizedPoints, outputClass, probability):

        val = normalizedPoints.mean.astype(int)
        cv2.circle(img, (val[0], val[1]), radius=3, color=(0,255,0), thickness=3)

        lmList = normalizedPoints.lmList

        roll, yaw, pitch = normalizedPoints.computeOrientation()
        normalizedPoints.computeDistanceWristMiddleFingerTip(pitch)
        # IF YOU WANT PRINT PITCH, ROLL, YAW
        #normalizedPoints.drawOrientationVector(img, roll, yaw, pitch)

        if "INIZIALIZATION" == self.currentState:
            # fill all the queue before START state
            if self.queueObj.isFullQueue():
                self.currentState = "START" #exit from INIZIALIZATION mode

            self.drawLog(img, (0,0,255), 0, val)

        elif "START" == self.currentState:
            x_mean, y_mean = self.queueObj.mean()
            cv2.circle(img, (x_mean,y_mean), radius=3, color=(255,0,0), thickness=3)
            checkStart = math.sqrt( (x_mean - val[0] )**2 + (y_mean - val[1])**2 )

            if checkStart < self.tolleranceSTART and self.queueObj.checkGesture("stop"):
                self.idxPoint = 0
                cv2.circle(img, (val[0], val[1]), radius=self.tolleranceSTART, color=(0,0,255), thickness=1) # draw the tollerance inside 
                cv2.circle(img, (val[0], val[1]), radius=self.tolleranceSTART+100, color=(0,255,0), thickness=1) # draw the tollerance outside 
                
                if len(self.traj.trajSpeed) == 0:
                    # mean of all distances from mean point val and hand landmark in lmList
                    self.previous_mean_distance = self.distanceFromMeanPoint(lmList, val)
                    self.traj.addPoint(x = val[0] / self.width,
                                       y = self.scale / 50,
                                       z = 1 - (val[1] / self.height),
                                       roll = roll,
                                       yaw = yaw,
                                       pitch = pitch)
                                       
                    self.traj.setSpeed(0) # speed is zero at the beginning
                else:
                    self.addTrajectoryPointAndSpeed(lmList, val, roll, yaw, pitch)

                self.traj.startTimeTraj = self.traj.previousTime # update the startTimeTraj until tracking state

            elif checkStart < self.tolleranceSTART and self.queueObj.checkGesture("thumbsup"): # execute last trajectory
                
                if len(self.trajCOMPLETE) > 0: # if trajectory it exists

                    if os.name == 'posix': # if linux system

                        if self.trajFlag:
                            # destory 3d figure
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
                        
                    elif os.name == 'nt': # if windows system

                        # just execute directly into the window

                        xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.trajCOMPLETE[-1].skipEveryNpointsFunc()

                        self.startingPoint = ( int( xdata[1] * self.height),
                                            int( (1- zdata[1]) * self.width) )
                
                        self.executeTrajectory(img, xdata, zdata)

                        tmpTime = time.time()
                        if tmpTime > self.previousTmpTime + self.delayToExecuteTrajectory and self.idxPoint < len(xdata)-1:
                            self.idxPoint += 1
                            self.previousTime = tmpTime

                else:

                    # there is no trajectory in queue...
                    self.cleanTraj()
                    self.currentState = "INIZIALIZATION"

            elif self.tolleranceSTART < checkStart < self.tolleranceSTART+100 and self.queueObj.checkGesture("stop"):
                self.traj.saveLastNValues(nPoints = 20) # take only the last 10 points
                xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.traj.skipEveryNpointsFunc()

                if len(xdata) > 0: # because otherwise could give index out of range
                    self.currentState = "TRACKING"
                    self.startingPoint = ( int( xdata[1] * self.height),
                                           int( (1- zdata[1]) * self.width) )

                self.addTrajectoryPointAndSpeed(lmList, val, roll, yaw, pitch)

                self.drawLog(img, (0,255,0), checkStart, val)
            else:
                self.drawLog(img, (0,0,255), checkStart, val)

        elif "TRACKING" == self.currentState:
            # il vantaggio di mediare sugli ultimi valori è perché in questo modo se viene perso il flusso lo si può riprendere molto velocemente
            # se si riparte dallo stesso punto in cui si è perso il tracking, diversamente bisognerà riacquisire tante nuove posizioni finchè 
            # la nuova media non sia sotto la tolleranza

            x_mean, y_mean = self.queueObj.meanOfTheLastNelements(self.nlastMovements)
            cv2.circle(img, (x_mean,y_mean), radius=3, color=(255,0,0), thickness=3)
            checkStartTracking = math.sqrt( (x_mean - val[0] )**2 + (y_mean - val[1])**2 )

            if self.queueObj.checkGesture("thumbsup"):
                self.currentState = "EXIT"

                # remove last n keypoint because the movemente to thumbup
                self.traj.thumbsUpFix(numberKeyPoints=10)

                currentTraj = copy.deepcopy(self.traj)
                self.trajCOMPLETE.append(currentTraj)

                # wait at least n seconds to catch another trajectory
                self.waitForNewTraj = time.time() + self.timeToCatchAnotherTraj

                # smooth every data
                xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.trajCOMPLETE[-1].skipEveryNpointsFunc()
                self.smoothing.setPoints(xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed)

                # this is useful otherwise there is overlap of old and new points
                if self.log3D:
                    self.drawTraj.clean()

            elif checkStartTracking < self.tolleranceTRACKING and self.queueObj.checkGesture("stop"):
                # mean of all distances from mean point val and hand landmark in lmList
                current_mean_dist = self.distanceFromMeanPoint(lmList, val)

                if current_mean_dist > self.previous_mean_distance:
                    self.scale += 1 * abs(current_mean_dist - self.previous_mean_distance) # greater if the difference is high
                else:
                    self.scale -= 1 * abs(current_mean_dist - self.previous_mean_distance) # same

                #print(self.scale)
                self.previous_mean_distance = current_mean_dist

                if self.traj.checkTrajTimeDuration():
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
                    self.traj.setSpeed(currentSpeed) # speed is zero at the beginning
                
            xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.traj.skipEveryNpointsFunc()
            self.draw2dTraj(img, xdata, zdata)

            if self.log3D:
                self.drawTraj.run(xdata, ydata, zdata, directionx, directiony, directionz, speed)

            self.drawLog(img, (255,0,0), checkStartTracking, val)


        elif "EXIT" == self.currentState:
            currentTime = time.time()
            if self.waitForNewTraj < currentTime and self.queueObj.checkGesture("stop"):
                self.cleanTraj()
                self.currentState = "INIZIALIZATION"
            
            #xdata, ydata, zdata, directionx, directiony, directionz, dtime, speed = self.trajCOMPLETE[-1].skipEveryNpointsFunc()
            xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed = self.smoothing.smoothCalculation()
            
            # export data as csv
            #pd.DataFrame(np.array([xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed])).to_csv("data.csv", index=False, header=None)
                        
            self.draw2dTraj(img, xdata, zdata)

            if self.log3D:
                self.drawTraj.run(xdata, ydata, zdata, rolldata, yawdata, pitchdata, speed)

            self.drawLog(img, (0,0,255), 0, val)

        self.queueObj.addMeanAndMatch(val, outputClass, probability)


def main():

   print("hello")


if __name__ == "__main__":
    
    main()