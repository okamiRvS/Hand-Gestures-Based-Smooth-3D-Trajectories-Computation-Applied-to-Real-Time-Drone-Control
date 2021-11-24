import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT


class trajectory():

    def __init__(self, skipEveryNpoints, trajTimeDuration):

        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []

        self.roll = []
        self.yaw = []
        self.pitch = []

        self.directionx = []
        self.directiony = []
        self.directionz = []

        self.trajSpeed = []

        self.skipEveryNpoints = skipEveryNpoints

        self.startTimeTraj = 0 # set when start tracking
        self.previousTime = self.startTimeTraj
        self.trajTimeDuration = trajTimeDuration


    def checkTrajTimeDuration(self):

        currentTime = time.time()
        if self.startTimeTraj + self.trajTimeDuration > currentTime:

            return True
        else:

            return False


    def addPoint(self, x, y, z, roll, yaw, pitch):

        self.trajPointsX.append(x)
        self.trajPointsY.append(y)
        self.trajPointsZ.append(z)
    
        self.roll.append(roll)
        self.yaw.append(yaw)
        self.pitch.append(pitch)

        self.computeDirection(x, y, z, roll, yaw, pitch)

    def computeDirection(self, x, y, z, roll, yaw, pitch):
        # this is vec=(1,0,0) in homogeneous coordinates

        vec = np.array([0,1,0,1])

        roll = -roll * np.pi / 180
        yaw = -yaw * np.pi / 180
        pitch = -pitch * np.pi / 180

        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        Matrix3dRotationX = np.array([[1, 0, 0, 0], [0, np.cos(pitch), np.sin(pitch), 0], [0, -np.sin(pitch), np.cos(pitch), 0], [0, 0, 0, 1]])
        Matrix3dRotationY = np.array([[np.cos(roll), 0, -np.sin(roll), 0], [0, 1, 0, 0], [np.sin(roll), 0, np.cos(roll), 0], [0, 0, 0, 1]])
        Matrix3dRotationZ = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], [np.sin(yaw), np.cos(yaw), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        vec = (Matrix3dRotationX @ vec.T).T
        vec = (Matrix3dRotationY @ vec.T).T
        vec = (Matrix3dRotationZ @ vec.T).T

        self.directionx.append(vec[0])
        self.directiony.append(vec[1])
        self.directionz.append(vec[2])


    def setSpeed(self, speed):

        self.trajSpeed.append(speed)

    def computeIstantSpeed(self):

        currentTime = time.time()
        deltaTime = currentTime - self.previousTime
        self.previousTime = currentTime

        try:
            distanceSpaceBetweenTwoLast3dPoints = math.sqrt( 
                ( self.trajPointsX[-2] - self.trajPointsX[-1] )**2 +
                ( self.trajPointsY[-2] - self.trajPointsY[-1] )**2 +
                ( self.trajPointsZ[-2] - self.trajPointsZ[-1] )**2
            )
        except:
            print("An exception occurred")
            distanceSpaceBetweenTwoLast3dPoints = 0 # this set currentSpeed to zero

        factorScale = 10
        currentSpeed = int(factorScale * distanceSpaceBetweenTwoLast3dPoints/deltaTime)

        return currentSpeed
    

    def reset(self):

        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []

        self.roll = []
        self.yaw = []
        self.pitch = []

        self.directionx = []
        self.directiony = []
        self.directionz = []

        self.trajSpeed = []


    def skipEveryNpointsFunc(self):

        # skip each n points, to have a better view of data
        xdata = self.trajPointsX[::self.skipEveryNpoints]
        ydata = self.trajPointsY[::self.skipEveryNpoints]
        zdata = self.trajPointsZ[::self.skipEveryNpoints]

        rolldata = self.roll[::self.skipEveryNpoints]
        yawdata = self.yaw[::self.skipEveryNpoints]
        pitchdata = self.pitch[::self.skipEveryNpoints]

        directionx = self.directionx[::self.skipEveryNpoints]
        directiony = self.directiony[::self.skipEveryNpoints]
        directionz = self.directionz[::self.skipEveryNpoints]

        speed = self.trajSpeed[::self.skipEveryNpoints]

        return xdata, ydata, zdata, directionx, directiony, directionz, speed


    def saveLastNValues(self, nPoints):

        lenTraj = len(self.trajPointsX) - 1
        takeOnly = lenTraj - nPoints 

        if takeOnly < 0:
            takeOnly = 0

        self.trajPointsX = self.trajPointsX[takeOnly:]
        self.trajPointsY = self.trajPointsY[takeOnly:]
        self.trajPointsZ = self.trajPointsZ[takeOnly:]

        self.roll = self.roll[takeOnly:]
        self.yaw = self.yaw[takeOnly:]
        self.pitch = self.pitch[takeOnly:]

        self.directionx = self.directionx[takeOnly:]
        self.directiony = self.directiony[takeOnly:]
        self.directionz = self.directionz[takeOnly:]

        self.trajSpeed = self.trajSpeed[takeOnly:]


    def thumbsUpFix(self, numberKeyPoints):

        # remove last n keypoint because the movemente to thumbup
        self.trajPointsX = self.trajPointsX[:-numberKeyPoints]
        self.trajPointsY = self.trajPointsY[:-numberKeyPoints]
        self.trajPointsZ = self.trajPointsZ[:-numberKeyPoints]

        self.roll = self.roll[:-numberKeyPoints]
        self.yaw = self.yaw[:-numberKeyPoints]
        self.pitch = self.pitch[:-numberKeyPoints]

        self.directionx = self.directionx[:-numberKeyPoints]
        self.directiony = self.directiony[:-numberKeyPoints]
        self.directionz = self.directionz[:-numberKeyPoints]

        self.trajSpeed = self.trajSpeed[:-numberKeyPoints]


def main():

   print("hello")


if __name__ == "__main__":
    
    main()