import numpy as np
import pdb


class pointManipulation():
    
    def __init__(self):

        self.height = 0
        self.width = 0


    def setSize(self, height: int, width: int):
        """
        Set dimension of picture.
        """ 

        self.height = height
        self.width = width


    def convertOriginBottomLeft(self, vector: np.array) -> np.array:
        """
        This function permits to move the origin from top left to bottom left.
        It is useful because opencv has a different way to express x/y coordinate.
        """

        vector[1] = self.height - vector[1]
        
        return vector


    def findAngle(self, vec1: np.array, vec2: np.array) -> float:
        """
        For the stop gesture find the angle that identify the pitch.
        """
 
        sin = vec1[0] - vec2[0]
        cos = vec1[1] - vec2[1]

        theta = np.arctan2(sin, cos)       
        #print("\narctan2 value : \n", theta * 180 / np.pi)

        return theta


    def translate(self, tmp: np.array, ty: float, tx: float) -> np.array: 
        """
        Translate tmp points of ty and tx position.
        """
        
        # build matrix 2d translation
        # https://ncase.me/matrix/
        Matrix2dTranslate = np.array([[1, 0, ty], [0, 1, tx], [0, 0, 1]])

        # apply the transformation to the vector
        tmp = (Matrix2dTranslate @ tmp.T).T

        return tmp


    def rotatate(self, tmp: np.array, theta: float) -> np.array:
        """
        Rotate tmp points of theta angle
        """

        # build matrix 2d rotation
        # https://ncase.me/matrix/
        Matrix2dRotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        # apply the transformation to the vector
        tmp = (Matrix2dRotation @ tmp.T).T

        return tmp

    def rotatate3D(self, tmp: np.array, roll, yaw, pitch) -> np.array:
        """
        Rotate tmp points of theta angle
        """

        # Matrix3dRotationX = np.array([[1, 0, 0, 0], [0, np.cos(roll), np.sin(roll), 0], [0, -np.sin(roll), np.cos(roll), 0], [0, 0, 0, 1]])
        # Matrix3dRotationY = np.array([[np.cos(yaw), 0, -np.sin(yaw), 0], [0, 1, 0, 0], [np.sin(yaw), 0, np.cos(yaw), 0], [0, 0, 0, 1]])
        # Matrix3dRotationZ = np.array([[np.cos(pitch), -np.sin(pitch), 0, 0], [np.sin(pitch), np.cos(pitch), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        tmp[:,2] = 0
        tmp = np.c_[tmp, np.ones(tmp.shape[0])]

        # Convert from radians in degree
        roll = roll * np.pi / 180 
        yaw = yaw * np.pi / 180 
        pitch = pitch * np.pi / 180 

        # Build matrix
        # https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html
        Matrix3dRotationX = np.array([[1, 0, 0, 0], [0, np.cos(pitch), np.sin(pitch), 0], [0, -np.sin(pitch), np.cos(pitch), 0], [0, 0, 0, 1]])
        Matrix3dRotationY = np.array([[np.cos(yaw), 0, -np.sin(yaw), 0], [0, 1, 0, 0], [np.sin(yaw), 0, np.cos(yaw), 0], [0, 0, 0, 1]])
        Matrix3dRotationZ = np.array([[np.cos(roll), -np.sin(roll), 0, 0], [np.sin(roll), np.cos(roll), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        # apply the transformation to the vector
        tmp = (Matrix3dRotationX @ Matrix3dRotationY @ Matrix3dRotationZ @ tmp.T).T

        tmp = tmp[:,:-1]
        tmp[:,-1] = 1

        return tmp


    def scaleMaxDistance(self, tmp: np.array) -> np.array:
        """
        Scale all tmp points wrt the max distance between centre of hand (detect gesture)
        and all the others hand landmarks.
        """

        distances = np.sqrt([tmp[:,0]**2 + tmp[:,1]**2])
        maxDistance = np.max(distances)
        tmp = tmp / maxDistance
        tmp[:,2] = np.ones( tmp.shape[0] ) # I can delete this, it's not useful, but maybe elegant...

        return tmp