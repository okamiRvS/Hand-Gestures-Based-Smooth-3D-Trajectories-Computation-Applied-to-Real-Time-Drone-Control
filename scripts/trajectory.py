import cv2
import numpy as np
import pdb
import math
import time
import dynamic3dDrawTrajectory as d3dT


class trajectory():

    def __init__(self, skipEveryNsec: float, trajTimeDuration: float):

        # Lists to save point (x,y,z)
        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []

        # Lists to save orientation (roll,yaw,pitch)
        self.roll = []
        self.yaw = []
        self.pitch = []

        # Lists to directions of each point
        self.directionx = []
        self.directiony = []
        self.directionz = []

        # List to save the time elapsed from beginnning each time a new point added
        self.dtime = []

        # List of norm speed computed each time a new point added
        self.trajSpeed = []

        # Do not add points if not passed at least self.skipEveryNsec
        self.skipEveryNsec = skipEveryNsec

        # Do not add points anymore if passed trajTimeDuration
        self.trajTimeDuration = trajTimeDuration
        
        # startTimeTraj will start when in TRACKING state
        self.startTimeTraj = -1 
        
        # currentTime value will be update each frame
        self.currentTime = -1
        
        # previousTime value will be update when passed "skipEveryNsec" secs
        self.previousTime = -1
        
        # startTime will start at the very beginning, in START state
        self.startTime = -1

        # Time elapsed from the current point added and the point before
        self.deltaTime = 0


    def checkTrajTimeDuration(self) -> bool:
        """
        Check if it is possibile to add other points into the trajectory 
        queue given the self.trajTimeDuration. The begining of check
        starts when TRACKING state is available.
        """

        # at the beginning 
        if self.startTimeTraj == -1:
            self.startTimeTraj = time.time()

        # print(currentTime - self.startTimeTraj > self.trajTimeDuration)
        # print(currentTime - self.startTimeTraj)

        if self.currentTime - self.startTimeTraj > self.trajTimeDuration:
            return False
        else:
            return True


    def checkIsPossibleAddPoint(self) -> bool:
        """
        Check if it is possibile to add other points into the trajectory
        queue given the self.skipEveryNsec. The beginning of check
        starts when START state is available. This check works also in
        TRACKING state. 
        """

        if self.startTime == -1:
            self.startTime = time.time()

        # at the beginning initialize the previousTime and startTime
        if self.previousTime == -1:
            self.previousTime = time.time()

        self.currentTime = time.time()

        # If passed 0.25 sec then return True to permit adding a point
        # and update previous time
        if self.currentTime - self.previousTime > self.skipEveryNsec:
            self.deltaTime = self.currentTime - self.previousTime
            self.previousTime = self.currentTime
            return True
        else:
            return False


    def addTimeElapsed(self):
        """
        Add time elapsed from self.startTime to the self.currentTime
        in the dtime list.
        """
        
        self.dtime.append(self.currentTime - self.startTime)


    def addPoint(self, x: float, y: float, z: float, roll: float, yaw: float, pitch: float):
        """
        Add position (x,y,z) and orientation (roll,yaw,pitch) in their 
        respective lists.
        Compute also direction using point and orientation.
        """
        
        self.trajPointsX.append(x)
        self.trajPointsY.append(y)
        self.trajPointsZ.append(z)
    
        self.roll.append(roll)
        self.yaw.append(yaw)
        self.pitch.append(pitch)

        self.computeDirection(x, y, z, roll, yaw, pitch)


    def computeDirection(self, x: float, y: float, z: float, roll: float, yaw: float, pitch: float):
        """
        Compute direction given a point (x,y,z) and the orientation. 
        """
        
        # this is vec=(1,0,0) in homogeneous coordinates
        vec = np.array([0,1,0,1])

        roll = -roll * np.pi / 180
        yaw = -yaw * np.pi / 180
        pitch = -pitch * np.pi / 180

        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        Matrix3dRotationX = np.array([[1, 0, 0, 0],
                                     [0, np.cos(pitch), np.sin(pitch), 0],
                                     [0, -np.sin(pitch), np.cos(pitch), 0],
                                     [0, 0, 0, 1]])
        Matrix3dRotationY = np.array([[np.cos(roll), 0, -np.sin(roll), 0],
                                     [0, 1, 0, 0],
                                     [np.sin(roll),0, np.cos(roll), 0],
                                     [0, 0, 0, 1]])
        Matrix3dRotationZ = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0], 
                                     [np.sin(yaw), np.cos(yaw), 0, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])

        vec = (Matrix3dRotationX @ vec.T).T
        vec = (Matrix3dRotationY @ vec.T).T
        vec = (Matrix3dRotationZ @ vec.T).T

        self.directionx.append(vec[0])
        self.directiony.append(vec[1])
        self.directionz.append(vec[2])


    def setSpeed(self, speed: float):
        """
        Add the norm of speed in the trajSpeed list.
        """
            
        self.trajSpeed.append(speed)


    def computeIstantSpeed(self) -> float:
        """
        Compute speed given the last two points added and the time self.deltaTime.
        """
        
        if self.deltaTime != 0:
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

            currentSpeed = int(factorScale * distanceSpaceBetweenTwoLast3dPoints / self.deltaTime)
        
        else:
            currentSpeed = 0.0

        return currentSpeed
    

    def reset(self):
        """
        This function permits to reset every component of the current object.
        """

        self.trajPointsX = []
        self.trajPointsY = []
        self.trajPointsZ = []

        self.roll = []
        self.yaw = []
        self.pitch = []

        self.directionx = []
        self.directiony = []
        self.directionz = []

        self.startTimeTraj = -1
        self.currentTime = -1
        self.previousTime = -1
        self.startTime = -1
        self.dtime = []

        self.trajSpeed = []


    def getData(self):
        """
        Return all object components.
        """

        return self.trajPointsX, self.trajPointsY, self.trajPointsZ, self.directionx, self.directiony, self.directionz, self.dtime, self.trajSpeed


    def saveLastNValues(self, nPoints: int):
        """
        This function is useful to take only the n frames during the START state,
        because otherwise we loose some part of trajectory
        """

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

        self.dtime = self.dtime[takeOnly:]

        self.trajSpeed = self.trajSpeed[takeOnly:]


    def thumbsUpFix(self, numberKeyPoints: int):
        """
        When we close the trajectory it is necessary a thumbs up gesture
        but closer to that gesture some points are detected and are noises, therefore
        we need to remove them (just few frames).
        """

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

        self.dtime = self.dtime[:-numberKeyPoints]

        self.trajSpeed = self.trajSpeed[:-numberKeyPoints]


def main():

   print("hello")


if __name__ == "__main__":
    
    main()