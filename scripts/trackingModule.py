import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT
import trajectory as traj
import copy

class tracking():

    def __init__(self, queueObj, skipEveryNpoints, trajTimeDuration):
        self.queueObj = queueObj 
        self.currentState = "INIZIALIZATION"
        self.tolleranceSTART = 80
        self.tolleranceTRACKING = 500 # 100 before
        self.nlastMovements = 5
        self.scale = 0
        self.previous_mean_distance = 0
        self.drawTraj = d3dT.dynamic3dDrawTrajectory()

        self.traj = traj.trajectory(skipEveryNpoints, trajTimeDuration)
        self.trajCOMPLETE = []

        self.height = 0
        self.width = 0

        self.timeToCatchAnotherTraj = 5

        self.idxPoint = 0
        self.delayToExecuteTrajectory = 50
        self.previousTmpTime = 0

    def drawLog(self, img, color, checkTollerance, val):
        fontScale = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        cv2.putText(img, f"checkTollerance: {checkTollerance:.2f}", (10,70), font, fontScale, color, thickness)
        cv2.putText(img, f"val: {val}", (10,100), font, fontScale, color, thickness)
        cv2.putText(img, f"scale: {self.scale}", (10,130), font, fontScale, color, thickness)
        cv2.putText(img, f"{self.currentState}", (10,160), font, fontScale, color, thickness)

    def draw2dTraj(self, img, xdata, zdata):
        numberOfPoints = len(xdata)

        for i in range(1, numberOfPoints):
            self.endingPoint = ( int(xdata[i] * self.height), int( (1-zdata[i]) * self.width) )
            cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
            cv2.circle(img, self.endingPoint, radius=0, color=(0,255,0), thickness=-1)
            
            if i != 1:
                cv2.line(img, self.startingPoint, self.endingPoint, (255,255,0), thickness=1)
            
            self.startingPoint = self.endingPoint

    def setSize(self, height, width):
        self.height = height
        self.width = width

    def justDrawLast2dTraj(self, img):
        xdata, ydata, zdata, speed = self.traj.skipEveryNpointsFunc()
        self.draw2dTraj(img, xdata, zdata)

    def distanceFromMeanPoint(self, lmList, val):
        # mean of all distances from mean point val and hand landmark in lmList to gain 3d scale factor
        # if it's greater respect of the previous istance than the hand is closer to the camera, otherwise is farther
        # the hand should be totally stopped
        sum_distances = 0
        for hand_land in lmList:
            sum_distances += math.sqrt( (val[0] - hand_land[1] )**2 + (val[1] - hand_land[2])**2 )

        return sum_distances / 21

    def addTrajectoryPointAndSpeed(self, lmList, val):
        # mean of all distances from mean point val and hand landmark in lmList
        current_mean_dist = self.distanceFromMeanPoint(lmList, val)

        if current_mean_dist > self.previous_mean_distance:
            self.scale += 1 * abs(current_mean_dist - self.previous_mean_distance) # greater if the difference is high
        else:
            self.scale -= 1 * abs(current_mean_dist - self.previous_mean_distance) # same

        print(self.scale)
        self.previous_mean_distance = current_mean_dist

        self.traj.addPoint(val[0] / self.height, self.scale / 50, 1 - (val[1] / self.width) )
        currentSpeed = self.traj.computeIstantSpeed() # compute istant speed
        self.traj.setSpeed(currentSpeed)

    def cleanTraj(self):
        # clean everything
        self.scale = 0
        self.previous_mean_distance = 0
        self.traj.reset()
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

    def orientationTest(self, p, q, r, tollerance):
        #testOnThisPhalanges = [[5,6,7], [6,7,8], [9,10,11], [10,11,12], [13,14,15], [14,15,16], [17,18,19], [18,19,20]]

        tmp = np.vstack((p,q))
        tmp = np.vstack((tmp,r))
        ones = np.ones(3)
        tmp[:,0] = ones
        tmp[:,2] = self.height - tmp[:,2] # to move the origin from top left to bottom left 

        res = np.linalg.det(tmp)

        # this is my version of orientation test with a tollerance
        orientation = None
        if res > 60:
            orientation = "back" #COUNTERclockwise
        elif res < -80:
            orientation = "front" #clockwise
        else:
            orientation = "center" #colinear

        return orientation

    def computeRoll(self, lmList):
        wrist = np.array(lmList[0], dtype=np.int32) # palmo
        middle_finger_tip = np.array(lmList[12], dtype=np.int32) # punta medio

        # compute the vector that pass at the center
        centerVector = middle_finger_tip - wrist
        
        # compute the vector that is perpendicular to the x axis and passes in wrist point
        # vec = np.array([wrist[1] + 100, wrist[2]], dtype=np.int32)
        # cv2.arrowedLine(img, centerVectorStart, (vec[0], vec[1]), (220, 25, 6), thickness=2, line_type=cv2.LINE_AA, shift=0, tipLength=0.3)
        
        # compute the inner product to know if orientation is left or right
        rightVector = np.array([1,0], dtype=np.int32)

        centerVectorNormalized = centerVector[1:] / np.sqrt(centerVector[1]**2 + centerVector[2]**2)
        res = np.inner(centerVectorNormalized, rightVector)

        # this is my version of inner product geometric interpretation with a tollerance
        roll = None
        if res > 0.3: # inner product 
            roll = "right"
        elif res < -0.3:
            roll = "left"
        else:
            roll = "center"s
        
        print(res)
        return roll

    def computePitch(self, lmList, pitch):

        if pitch == "right":
            tollerance = 0
            p = np.array(lmList[9], dtype=np.int32) 
            q = np.array(lmList[10], dtype=np.int32)
            r = np.array(lmList[11], dtype=np.int32)

            return self.orientationTest(p, q, r, tollerance)

        elif pitch == "left":
            tollerance = 1
            p = np.array(lmList[9], dtype=np.int32) 
            q = np.array(lmList[10], dtype=np.int32)
            r = np.array(lmList[11], dtype=np.int32)

            return self.orientationTest(r, q, p, tollerance)

        elif pitch == "center":
            if lmList[4][2] < lmList[6][2]:
                return "back"
            elif lmList[4][2] > lmList[5][2]:
                return "front"
            else:
                return "center"

    def drawOrientationVector(self, img, lmList, pitch, roll):
        wrist = np.array(lmList[0], dtype=np.int32) # palmo
        middle_finger_tip = np.array(lmList[12], dtype=np.int32) # punta medio

        # compute the vector that pass at the center
        centerVector = 1.2 * ( middle_finger_tip - wrist )

        centerVectorEnd = wrist + centerVector
        centerVectorEnd = ( int(centerVectorEnd[1]), int(centerVectorEnd[2]) )
        centerVectorStart = (wrist[1], wrist[2])
        cv2.arrowedLine(img, centerVectorStart, centerVectorEnd, (220, 25, 6), thickness=2, line_type=cv2.LINE_AA, shift=0, tipLength=0.3)

        fontScale = 0.6
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 2
        cv2.putText(img, f"Roll: {roll}, Pitch: {pitch}", (centerVectorEnd[0]+20,centerVectorEnd[1]), font, fontScale, (0, 225, 0), thickness)

    def run(self, img, lmList, outputClass, probability):

        # mean x and y of all hand leandmark
        x_sum = y_sum = 0
        for val in lmList:
            x_sum += val[1]
            y_sum += val[2]
        
        x_mean = x_sum / 21
        y_mean = y_sum / 21
        val = np.array([x_mean, y_mean], dtype=np.int32)
        cv2.circle(img, (val[0], val[1]), radius=3, color=(0,255,0), thickness=3)

        roll = self.computeRoll(lmList)
        pitch = self.computePitch(lmList, roll)
        self.drawOrientationVector(img, lmList, pitch, roll)

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

                    self.traj.addPoint(val[0] / self.height, self.scale / 50, 1 - (val[1] / self.width))
                    self.traj.setSpeed(0) # speed is zero at the beginning
                else:
                    self.addTrajectoryPointAndSpeed(lmList, val)

                self.traj.startTimeTraj = self.traj.previousTime # update the startTimeTraj until tracking state

            elif checkStart < self.tolleranceSTART and self.queueObj.checkGesture("thumbsup"): # execute last trajectory
                
                if len(self.trajCOMPLETE) > 0:

                    xdata, ydata, zdata, speed = self.trajCOMPLETE[-1].skipEveryNpointsFunc()

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
                xdata, ydata, zdata, speed = self.traj.skipEveryNpointsFunc()

                if len(xdata) > 0: # because otherwise could give index out of range
                    self.currentState = "TRACKING"
                    self.startingPoint = ( int( xdata[1] * self.height),
                                           int( (1- zdata[1]) * self.width) )

                self.addTrajectoryPointAndSpeed(lmList, val)

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
                self.traj.thumbsUpFix(numberKeyPoints=30)

                currentTraj = copy.deepcopy(self.traj)
                self.trajCOMPLETE.append(currentTraj)

                # wait at least n seconds to catch another trajectory
                self.waitForNewTraj = time.time() + self.timeToCatchAnotherTraj

            elif checkStartTracking < self.tolleranceTRACKING and self.queueObj.checkGesture("stop"):
                # mean of all distances from mean point val and hand landmark in lmList
                current_mean_dist = self.distanceFromMeanPoint(lmList, val)

                if current_mean_dist > self.previous_mean_distance:
                    self.scale += 1 * abs(current_mean_dist - self.previous_mean_distance) # greater if the difference is high
                else:
                    self.scale -= 1 * abs(current_mean_dist - self.previous_mean_distance) # same

                print(self.scale)
                self.previous_mean_distance = current_mean_dist

                if self.traj.checkTrajTimeDuration():
                    # collect data to draw the 3d trajectory
                    # scale X,Z data from 0 to 1; about scale factor I consider 50 values, but maybe it requires some major details...
                    self.traj.addPoint(val[0] / self.height, self.scale / 50, 1 - (val[1] / self.width) )
                    
                    # compute istant speed
                    currentSpeed = self.traj.computeIstantSpeed()
                    self.traj.setSpeed(currentSpeed) # speed is zero at the beginning
                
            xdata, ydata, zdata, speed = self.traj.skipEveryNpointsFunc()
            self.draw2dTraj(img, xdata, zdata)
            self.drawTraj.run(xdata, ydata, zdata, speed)

            self.drawLog(img, (255,0,0), checkStartTracking, val)


        elif "EXIT" == self.currentState:
            currentTime = time.time()
            if self.waitForNewTraj < currentTime and self.queueObj.checkGesture("stop"):
                self.cleanTraj()
                self.currentState = "INIZIALIZATION"
            
            xdata, ydata, zdata, speed = self.trajCOMPLETE[-1].skipEveryNpointsFunc()
            self.draw2dTraj(img, xdata, zdata)
            self.drawTraj.run(xdata, ydata, zdata, speed)
            self.drawLog(img, (0,0,255), 0, val)

        self.queueObj.addMeanAndMatch(val, outputClass, probability)


def main():
   print("hello")

if __name__ == "__main__":
    main()