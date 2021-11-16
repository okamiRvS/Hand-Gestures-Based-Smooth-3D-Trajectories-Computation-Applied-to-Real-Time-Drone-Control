import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT
import trajectory as traj

class tracking():

    def __init__(self, queueObj, skipEveryNpoints):
        self.queueObj = queueObj 
        self.currentState = "INIZIALIZATION"
        self.tolleranceSTART = 2
        self.tolleranceTRACKING = 500 # 100 before
        self.nlastMovements = 5
        self.scale = 0
        self.mean_distance = 0
        self.drawTraj = d3dT.dynamic3dDrawTrajectory()

        self.traj = traj.trajectory(skipEveryNpoints)

        self.height = 0
        self.width = 0

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

        count = 3
        if numberOfPoints == 2:
            self.startingPoint = ( int(xdata[1] * self.height), int( (1-zdata[1]) * self.width) )
            cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
        elif numberOfPoints > 2:
            for i in range(1,numberOfPoints):
                self.endingPoint = ( int(xdata[i] * self.height), int( (1-zdata[i]) * self.width) )
                cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
                cv2.circle(img, self.endingPoint, radius=0, color=(0,255,0), thickness=-1)
                
                if i != 1:
                    cv2.line(img, self.startingPoint, self.endingPoint, (255,255,0), thickness=1)
                
                self.startingPoint = self.endingPoint
                count+=1

    def setSize(self, height, width):
        self.height = height
        self.width = width

    def run(self, img, lmList):

        # mean x and y of all hand leandmark
        x_sum = y_sum = 0
        for val in lmList:
            x_sum += val[1]
            y_sum += val[2]
        
        x_mean = x_sum / 21
        y_mean = y_sum / 21
        val = np.array([x_mean, y_mean], dtype=np.int32)
        val_mean_point = (val[0], val[1])
        cv2.circle(img, val_mean_point, radius=3, color=(0,255,0), thickness=3)

        if "INIZIALIZATION" == self.currentState:
            # fill all the queue before start state
            if self.queueObj.isFullQueue():
                self.currentState = "START" #exit from INIZIALIZATION mode

            self.drawLog(img, (0,0,255), 0, val)

        elif "START" == self.currentState:
            x_mean, y_mean = self.queueObj.mean()
            cv2.circle(img, (x_mean,y_mean), radius=3, color=(255,0,0), thickness=3)
            checkStart = math.sqrt( (x_mean - val[0] )**2 + (y_mean - val[1])**2 )

            if checkStart < self.tolleranceSTART:
                self.currentState = "TRACKING"

                # mean of all distances from mean point val and hand landmark in lmList to gain 3d scale factor
                # if it's greater respect of the previous istance than the hand is closer to the camera, otherwise is farther
                # the hand should be totally stopped
                sum_distances = 0
                for hand_land in lmList:
                    sum_distances += math.sqrt( (val[0] - hand_land[1] )**2 + (val[1] - hand_land[2])**2 )

                self.mean_distance = sum_distances / 21

                self.traj.addPoint(val[0] / self.height, self.scale / 50, 1 - (val[1] / self.width))
                self.traj.setSpeed(0) # speed is zero at the beginning

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

            if checkStartTracking < self.tolleranceTRACKING:

                # mean of all distances from mean point val and hand landmark in lmList to gain 3d scale factor
                # if it's greater respect of the previous istance than the hand is closer to the camera, otherwise is farther
                # the hand should be totally stopped
                sum_distances = 0
                for hand_land in lmList:
                    sum_distances += math.sqrt( (val[0] - hand_land[1] )**2 + (val[1] - hand_land[2])**2 )

                tmp_mean_dist = sum_distances / 21

                if tmp_mean_dist > self.mean_distance:
                    self.scale+=1
                else:
                    self.scale-=1

                self.mean_distance = tmp_mean_dist

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
            else:
                # clean everything
                self.drawLog(img, (0,0,255), checkStartTracking, val)
                self.scale = 0
                self.mean_distance = 0
                self.traj.reset()
                self.drawTraj.clean()
                self.currentState = "START"

        self.queueObj.add(val)


def main():
   print("hello")

if __name__ == "__main__":
    main()