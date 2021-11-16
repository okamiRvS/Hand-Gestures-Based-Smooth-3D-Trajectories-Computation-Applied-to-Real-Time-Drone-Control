import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT

class trajectory():

    def __init__(self, skipEveryNpoints):

        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []
        self.trajSpeed = []

        self.skipEveryNpoints = skipEveryNpoints
        self.previousTime = time.time()

    def addPoint(self, x, y, z):
        self.trajPointsX.append(x)
        self.trajPointsY.append(y)
        self.trajPointsZ.append(z)
    
    def setSpeed(self, speed):
        self.trajSpeed.append(speed)

    def computeIstantSpeed(self):
        tmpTime = time.time()
        deltaTime = tmpTime - self.previousTime
        self.previousTime = tmpTime
        distanceSpaceBetweenTwoLast3dPoints = math.sqrt( 
            ( self.trajPointsX[-2] - self.trajPointsX[-1] )**2 +
            ( self.trajPointsY[-2] - self.trajPointsY[-1] )**2 +
            ( self.trajPointsZ[-2] - self.trajPointsZ[-1] )**2
        )
        factorScale = 10
        currentSpeed = int(factorScale * distanceSpaceBetweenTwoLast3dPoints/deltaTime)
        print(currentSpeed)
        return currentSpeed
    
    def reset(self):
        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []
        self.trajSpeed = []

    def skipEveryNpointsFunc(self):
        # skip each n points, to have a better view of data
        xdata = self.trajPointsX[::self.skipEveryNpoints]
        ydata = self.trajPointsY[::self.skipEveryNpoints]
        zdata = self.trajPointsZ[::self.skipEveryNpoints]
        speed = self.trajSpeed[::self.skipEveryNpoints]
        return xdata, ydata, zdata, speed


def main():
   print("hello")

if __name__ == "__main__":
    main()