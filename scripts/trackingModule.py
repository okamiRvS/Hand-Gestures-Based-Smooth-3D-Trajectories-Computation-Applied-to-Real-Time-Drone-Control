import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT

class tracking():

    def __init__(self, queueObj):
        self.queueObj = queueObj 
        self.currentState = "INIZIALIZATION"
        self.tolleranceSTART = 2
        self.tolleranceTRACKING = 100
        self.nlastMovements = 5
        self.scale = 0
        self.mean_distance = 0
        self.drawTraj = d3dT.dynamic3dDrawTrajectory()

        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []

        self.previousTime = time.time()
        self.trajSpeed = []

    def drawLog(self, img, color, checkTollerance, val):
        scale = 2
        cv2.putText(img, f"checkTollerance: {checkTollerance}", (200,50), cv2.FONT_HERSHEY_PLAIN, scale, color, 3)
        cv2.putText(img, f"val: {val}", (200,100), cv2.FONT_HERSHEY_PLAIN, scale, color, 3)
        cv2.putText(img, f"scale: {self.scale}", (200,150), cv2.FONT_HERSHEY_PLAIN, scale, color, 3)
        cv2.putText(img, f"{self.currentState}", (200,200), cv2.FONT_HERSHEY_PLAIN, scale, color, 3)

    def run(self, img, lmList):
        height, width, _ = img.shape

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
                self.startingPoint = val_mean_point
                self.currentState = "TRACKING"

                # mean of all distances from mean point val and hand landmark in lmList to gain 3d scale factor
                # if it's greater respect of the previous istance than the hand is closer to the camera, otherwise is farther
                # the hand should be totally stopped
                sum_distances = 0
                for hand_land in lmList:
                    sum_distances += math.sqrt( (val[0] - hand_land[1] )**2 + (val[1] - hand_land[2])**2 )

                self.mean_distance = sum_distances / 21

                self.trajPointsX.append(val[0] / height)
                self.trajPointsY.append(self.scale / 100)
                self.trajPointsZ.append(val[1] / width)
                self.trajSpeed.append(0) # speed is zero at the beginning

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

            endingPoint = val_mean_point

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

                # draw the the trajectory
                '''
                cv2.circle(img, self.startingPoint, radius=0, color=(0,255,0), thickness=-1)
                cv2.circle(img, endingPoint, radius=0, color=(0,255,0), thickness=-1)
                cv2.line(img, self.startingPoint, endingPoint, (255,255,0), thickness=2)
                '''
                self.trajPointsX.append(val[0] / height)
                self.trajPointsY.append(self.scale / 100)
                self.trajPointsZ.append(1- (val[1] / width) )

                # compute istant speed
                deltaTime = time.time() - self.previousTime
                distanceSpaceBetweenTwoLast3dPoints = math.sqrt( 
                    ( self.trajPointsX[-2] - self.trajPointsX[-1] )**2 +
                    ( self.trajPointsY[-2] - self.trajPointsY[-1] )**2 +
                    ( self.trajPointsZ[-2] - self.trajPointsZ[-1] )**2
                )
                factor = 200000
                self.trajSpeed.append(factor * distanceSpaceBetweenTwoLast3dPoints/deltaTime)
                self.drawTraj.run(self.trajPointsX, self.trajPointsY, self.trajPointsZ, self.trajSpeed)

                self.drawLog(img, (255,0,0), checkStartTracking, val)
            else:
                self.drawLog(img, (0,0,255), checkStartTracking, val)
                self.scale = 0
                self.mean_distance = 0
                self.trajPointsX = []
                self.trajPointsY = []
                self.trajPointsZ = []
                self.trajSpeed = []
                self.drawTraj.clean()
                self.currentState = "START"

        self.queueObj.add(val)


def main():
   print("hello")

if __name__ == "__main__":
    main()