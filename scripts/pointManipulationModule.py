import numpy as np
   
class pointManipulation():
    
    def __init__(self):

        self.height = 0
        self.width = 0

    def setSize(self, height, width):

        self.height = height
        self.width = width

    def convertOriginBottomLeft(self, vector):

        # move the origin from top left to bottom left
        vector[1] = self.height - vector[1]
        
        return vector

    def findAngle(self, vec1, vec2):
 
        sin = vec1[0] - vec2[0]
        cos = vec1[1] - vec2[1]

        theta = np.arctan2(sin, cos)       
        #print("\narctan2 value : \n", theta * 180 / np.pi)

        return theta

    def translate(self, tmp, ty, tx): 
        
        # build matrix 2d translation
        # https://ncase.me/matrix/
        Matrix2dTranslate = np.array([[1, 0, ty], [0, 1, tx], [0, 0, 1]])

        # apply the transformation to the vector
        tmp = (Matrix2dTranslate @ tmp.T).T

        return tmp

    def rotatate(self, tmp, theta):

        # build matrix 2d rotation
        # https://ncase.me/matrix/
        Matrix2dRotation = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

        # apply the transformation to the vector
        tmp = (Matrix2dRotation @ tmp.T).T

        return tmp

    def scaleMaxDistance(self, tmp):

        distances = np.sqrt([tmp[:,0]**2 + tmp[:,1]**2])
        maxDistance = np.max(distances)
        tmp = tmp / maxDistance
        tmp[:,2] = np.ones( tmp.shape[0] ) # I can delete this, it's not useful, but maybe elegant...

        return tmp