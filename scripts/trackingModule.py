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

    def orientationTest(self, p, q, r, tol1, tol2, mean):
        #testOnThisPhalanges = [[5,6,7], [6,7,8], [9,10,11], [10,11,12], [13,14,15], [14,15,16], [17,18,19], [18,19,20]]

        tmp = np.vstack((p,q))
        tmp = np.vstack((tmp,r))
        
        ones = np.ones(3)
        tmp[:,0] = ones

        # to move the origin from top left to bottom left
        tmp[:,2] = self.height - tmp[:,2]  
        mean = np.array([ [mean[0], self.height -  mean[1] ] ], dtype=np.float32)

        # since mean point as anchor translate everything to the origin
        tmp[:,1:] = tmp[:,1:] - mean

        # scale everything respect max distance
        maxModule = np.max( np.sqrt([tmp[:,1]**2 + tmp[:,2]**2]) )
        tmp = tmp / maxModule
        tmp[:,0] = ones

        # # normalize each vector to 1 (except firt column)
        # #tmp[:,1:] = tmp[:,1:] / np.sqrt([tmp[:,1]**2 + tmp[:,2]**2]).T
        # #res = np.linalg.det(tmp)
        # #print(res*1000000)
        
        # res = np.linalg.det(tmp)

        # # this is my version of orientation test with a tollerance
        # orientation = None
        # if res > 150 : #60
        #     orientation = "-90" #COUNTERclockwise
        # elif res < -250: #-80
        #     orientation = "+90" #clockwise
        # else:
        #     orientation = "0" #colinear

        # return orientation

        res = np.linalg.det(tmp)
        return -res*1125 # 1125 is empirically computed

    def computeRoll(self, lmList):
        wrist = np.array(lmList[0], dtype=np.int32) # palmo
        middle_finger_tip = np.array(lmList[12], dtype=np.int32) # punta medio

        # # compute the vector that pass at the center
        # centerVector = middle_finger_tip - wrist
        
        # # compute the inner product to know if orientation is left or right
        # rightVector = np.array([1,0], dtype=np.int32)

        # centerVectorNormalized = centerVector[1:] / np.sqrt(centerVector[1]**2 + centerVector[2]**2)
        # res = np.inner(centerVectorNormalized, rightVector)      

        # # this is my version of inner product geometric interpretation with a tollerance
        # roll = None

        # if res > 0.3: # inner product 
        #     roll = "-90" # right
        # elif res < -0.3:
        #     roll = "+90" # left
        # else:
        #     roll = "0" # center
        # return roll

        # to move the origin from top left to bottom left 
        middle_finger_tip[2] = self.height - middle_finger_tip[2] 
        wrist[2] = self.height - wrist[2] 

        sin = middle_finger_tip[1] - wrist[1]
        cos = middle_finger_tip[2] - wrist[2]

        thetarad = np.arctan2(sin, cos)       
        #print("\narctan2 value : \n", thetarad * 180 / np.pi)
        thetadeg = thetarad * 180 / np.pi
        return -thetadeg


    def computeYaw(self, lmList, roll, mean):

        if roll < - 5: # "-90"
            tol1 = 150
            tol2 = -250
            p = np.array(lmList[5], dtype=np.float32)
            q = np.array(lmList[6], dtype=np.float32)
            r = np.array(lmList[7], dtype=np.float32)

        elif roll > 5: # "+90"
            tol1 = 150
            tol2 = -250
            p = np.array(lmList[5], dtype=np.float32) 
            q = np.array(lmList[6], dtype=np.float32)
            r = np.array(lmList[7], dtype=np.float32)

        else: # "0"
            tol1 = 150
            tol2 = -250
            p = np.array(lmList[9], dtype=np.float32) 
            q = np.array(lmList[10], dtype=np.float32)
            r = np.array(lmList[11], dtype=np.float32)

        return self.orientationTest(p, q, r, tol1, tol2, mean) / 3 # 3 is empirical, should be varied respect the distance

    def computePitch(self, lmList, roll, yaw, mean):
        # se la punta del pollice supera un punto della mano...
        # if -5 < roll < 5: # "0"
        #     if lmList[4][2] < lmList[6][2]: # HEY! HERE IS MINOR BECAUSE Y HAS DIFFERENT REFERENCE SYSTEM
        #         return "+90" # fron
        #     elif lmList[4][2] > lmList[5][2]:
        #         return "-90" # back
        #     else:
        #         return "0" # center

        middle_finger_tip = np.array(lmList[12], dtype=np.int32)
        wrist = np.array(lmList[0], dtype=np.int32) 

        # to move the origin from top left to bottom left 
        middle_finger_tip[2] = self.height - middle_finger_tip[2] 
        wrist[2] = self.height - wrist[2] 
        mean = np.array([ [mean[0], self.height -  mean[1] ] ], dtype=np.float32)

        # build matrix 2d rotation
        # https://ncase.me/matrix/ 
        sin = middle_finger_tip[1] - wrist[1]
        cos = middle_finger_tip[2] - wrist[2]

        theta = np.arctan2(sin, cos)       
        print("\narctan2 value : \n", theta * 180 / np.pi)
        
        Matrix2dRotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        # manipultate the vector to do the correct transformation
        thumb_tip = np.array([lmList[4][1], self.height - lmList[4][2], 1], dtype=np.int32)
        index_finger_mcp = np.array([lmList[5][1], self.height - lmList[5][2], 1], dtype=np.int32)
        index_finger_pip = np.array([lmList[6][1], self.height - lmList[6][2], 1], dtype=np.int32)

        # apply the transformation to the vector
        thumb_tip_tansf = Matrix2dRotation @ thumb_tip
        index_finger_mcp_trans = Matrix2dRotation @ index_finger_mcp
        index_finger_pip_trans = Matrix2dRotation @ index_finger_pip

        # create array
        tmp = np.array([thumb_tip_tansf, index_finger_mcp_trans, index_finger_pip_trans])

        # since mean point as anchor translate everything to the origin
        tmp[:,:-1] = tmp[:,:-1] - mean

        # scale everything respect max distance
        maxModule = np.max( np.sqrt([tmp[:,0]**2 + tmp[:,1]**2]) )
        tmp = tmp / maxModule
        tmp[:,2] = np.ones(3) # I can delete this, it's not useful, but maybe elegant...

        pointZero = (tmp[1,1] + tmp[2,1]) / 2
        factNormalized = pointZero - tmp[0,1]
        return -factNormalized*112 # 112 is empirically computed

    def drawOrientationVector(self, img, lmList, roll, yaw, pitch):
        wrist = np.array(lmList[0], dtype=np.int32) # palmo
        middle_finger_tip = np.array(lmList[12], dtype=np.int32) # punta medio

        # compute the vector that pass at the center
        centerVector = 1.2 * ( middle_finger_tip - wrist )

        centerVectorEnd = wrist + centerVector
        centerVectorEnd = ( int(centerVectorEnd[1]), int(centerVectorEnd[2]) )
        centerVectorStart = (wrist[1], wrist[2])
        cv2.arrowedLine(img, centerVectorStart, centerVectorEnd, (220, 25, 6), thickness=2, line_type=cv2.LINE_AA, shift=0, tipLength=0.3)

        fontScale = 0.5
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 2
        cv2.putText(img, f"Roll: {roll}", (centerVectorEnd[0]+20,centerVectorEnd[1]), font, fontScale, (0, 225, 0), thickness)
        cv2.putText(img, f"Yaw: {yaw}", (centerVectorEnd[0]+20,centerVectorEnd[1]+40), font, fontScale, (0, 225, 0), thickness)
        cv2.putText(img, f"Pitch: {pitch}", (centerVectorEnd[0]+20,centerVectorEnd[1]+80), font, fontScale, (0, 225, 0), thickness)

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
        yaw = self.computeYaw(lmList, roll, val) # ATTENTION REMEMBER THAT THERE IS ALSO THE YAW I NEED TO COMPUTE IT
        pitch = self.computePitch(lmList, roll, yaw, val)
        self.drawOrientationVector(img, lmList, roll, yaw, pitch)

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