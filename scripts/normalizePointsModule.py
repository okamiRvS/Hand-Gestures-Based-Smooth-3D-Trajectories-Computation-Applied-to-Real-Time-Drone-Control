import numpy as np
import pointManipulationModule as pm
import cv2
import copy
import pdb

class normalizePoints():

    def __init__(self):

        self.transf = pm.pointManipulation()

        # initialize array
        self.tmp = np.zeros((21,2), dtype=np.float32)

        self.mean = 0
        self.lmList = []

        self.height = 0
        self.width = 0


    def setSize(self, height, width):

        self.transf.setSize(height, width)
        self.height = height
        self.width = width


    def setArray(self, lmList):

        # mean x and y of all hand leandmark
        # assign value
        self.lmList = lmList # you should check if more than hand detected...
        tmp = self.tmp
        x_sum = y_sum = 0

        for i in range(len(lmList)):
            # create numpy array point
            tmp[i] = np.array(lmList[i][1:], dtype=np.float32)

            x_sum += tmp[i][0]
            y_sum += tmp[i][1]

            # to move the origin from top left to bottom left 
            tmp[i] = self.transf.convertOriginBottomLeft(tmp[i])

        self.tmp = tmp
       
        x_mean = x_sum / 21
        y_mean = y_sum / 21
 
        mean = np.array([x_mean, y_mean], dtype=np.float32)
        self.mean = mean


    def normalize(self):

        mean = copy.deepcopy(self.mean) # LASCIA COSì! DOBBIAMO COPIARE SELF.MEAN
        mean = self.transf.convertOriginBottomLeft(mean)

        # find angle
        theta = self.transf.findAngle(self.tmp[12], self.tmp[0])
        self.theta = theta

        # convert in homogeneous coordinates
        tmp = np.hstack( (self.tmp, np.ones((21,1)) ))

        # since mean point as anchor translate everything to the origin
        tmp = self.transf.translate(tmp, -mean[0], -mean[1])

        # compute rotation
        tmp = self.transf.rotatate(tmp, theta)

        # scale everything respect max distance
        tmp = self.transf.scaleMaxDistance(tmp)     

        # save this information
        self.tmp = tmp


    def computeOrientation(self):

        roll = self.computeRoll()
        yaw = self.computeYaw(roll)
        pitch = self.computePitch()

        return roll, yaw, pitch


    def computeRoll(self):

        thetadeg = self.theta * 180 / np.pi

        return -thetadeg


    def computeYaw(self, roll):

        if roll < - 5: # "-90"
            tol1 = 150
            tol2 = -250
            p = self.tmp[5]
            q = self.tmp[6]
            r = self.tmp[7]

        elif roll > 5: # "+90"
            tol1 = 150
            tol2 = -250
            p = self.tmp[5]
            q = self.tmp[6]
            r = self.tmp[7]

        else: # "0"
            tol1 = 150
            tol2 = -250
            p = self.tmp[9]
            q = self.tmp[10]
            r = self.tmp[11]

        return self.orientationTest(p, q, r, tol1, tol2)


    def orientationTest(self, p, q, r, tol1, tol2):

        #testOnThisPhalanges = [[5,6,7], [6,7,8], [9,10,11], [10,11,12], [13,14,15], [14,15,16], [17,18,19], [18,19,20]]

        tmp = np.vstack( (p,q) )
        tmp = np.vstack((tmp,r))
        ones = np.ones( (3,1) )
        tmp = np.hstack( (ones,tmp) )

        res = np.linalg.det(tmp)

       # this is a quadratic form, more stable to zero
        if res < 0:
            res = (res**2) / 1666 # 180 is empirically computed
        else:
            res = -(res**2) / 1666

        # the part above 90 degrees scale a lot
        if res < - 90:
            res = -90 - (res + 90) * 0.1
        elif res > 90:
            res = 90 + (res - 90) * 0.1
            
        return res


    def computePitch(self):

        thumb_tip  = self.tmp[4]
        index_finger_mcp = self.tmp[5]
        index_finger_pip = self.tmp[6]

        # copmute the difference from the mean between index_finger_mcp and index_finger_pip with the thumb_tip y value
        pointZero = (index_finger_mcp[1] + index_finger_pip[1]) / 2
        res = pointZero - thumb_tip[1]

        # this is a quadratic form, more stable to zero
        if res < 0:
            res = (res**2) / 31 # 180 is empirically computed
        else:
            res = -(res**2) / 31

        # the part above 90 degrees scale a lot
        if res < - 90:
            res = -90 - (res + 90) * 0.1
        elif res > 90:
            res = 90 + (res - 90) * 0.1

        return res

    
    def drawOrientationVector(self, img, roll, yaw, pitch):

        wrist = np.array(self.lmList[0], dtype=np.int32) # palmo
        middle_finger_tip = np.array(self.lmList[12], dtype=np.int32) # punta medio

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


    def drawAllHandTransformed(self, img):
        # scale a bit to draw points on canvas
        tmp = self.tmp
        tmp[:,:-1] = tmp[:,:-1] * 100
        tmp = self.transf.translate(tmp, 100, 130) #tmp + mean + np.array([-300, 0])
        tmp = tmp[:,:-1]
        tmp[:,1] = self.height - tmp[:,1]
        tmp = tmp.astype(int)

        # drawPoint
        fontScale = 0.3
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 1
        color = (255,255,0)
        for i in range(tmp.shape[0]):
            position = tuple(tmp[i])
            cv2.circle(img, position, radius=0, color=color, thickness=5)
            cv2.putText(img, str(i), (position[0]+10, position[1]), font, fontScale, color, thickness)

        # put text on thumb_tip, index_finger_mcp and index_finger_pip
        color = (0,255,0)
        position = tuple(tmp[4])
        cv2.circle(img, position, radius=0, color=color, thickness=5)
        cv2.putText(img, "thumb_tip", ( position[0] + 10, position[1] ), font, fontScale, color, thickness)
        position = tuple(tmp[5])
        cv2.circle(img, position, radius=0, color=color, thickness=5)
        cv2.putText(img, "index_finger_mcp", ( position[0] + 10, position[1] ), font, fontScale, color, thickness)
        position = tuple(tmp[6])
        cv2.circle(img, position, radius=0, color=color, thickness=5)
        cv2.putText(img, "index_finger_pip", ( position[0] + 10, position[1] ), font, fontScale, color, thickness)

        # connect points to get the hand shape
        color = (0,0,255)
        cv2.line(img, tuple(tmp[0]), tuple(tmp[1]), color, thickness=1)
        cv2.line(img, tuple(tmp[0]), tuple(tmp[5]), color, thickness=1)
        cv2.line(img, tuple(tmp[0]), tuple(tmp[17]), color, thickness=1)
        cv2.line(img, tuple(tmp[1]), tuple(tmp[2]), color, thickness=1)
        cv2.line(img, tuple(tmp[2]), tuple(tmp[3]), color, thickness=1)
        cv2.line(img, tuple(tmp[3]), tuple(tmp[4]), color, thickness=1)
        cv2.line(img, tuple(tmp[5]), tuple(tmp[6]), color, thickness=1)
        cv2.line(img, tuple(tmp[5]), tuple(tmp[9]), color, thickness=1)
        cv2.line(img, tuple(tmp[6]), tuple(tmp[7]), color, thickness=1)
        cv2.line(img, tuple(tmp[7]), tuple(tmp[8]), color, thickness=1)
        cv2.line(img, tuple(tmp[9]), tuple(tmp[10]), color, thickness=1)
        cv2.line(img, tuple(tmp[9]), tuple(tmp[13]), color, thickness=1)
        cv2.line(img, tuple(tmp[10]), tuple(tmp[11]), color, thickness=1)
        cv2.line(img, tuple(tmp[11]), tuple(tmp[12]), color, thickness=1)
        cv2.line(img, tuple(tmp[13]), tuple(tmp[14]), color, thickness=1)
        cv2.line(img, tuple(tmp[14]), tuple(tmp[15]), color, thickness=1)
        cv2.line(img, tuple(tmp[15]), tuple(tmp[16]), color, thickness=1)
        cv2.line(img, tuple(tmp[13]), tuple(tmp[17]), color, thickness=1)
        cv2.line(img, tuple(tmp[17]), tuple(tmp[18]), color, thickness=1)
        cv2.line(img, tuple(tmp[18]), tuple(tmp[19]), color, thickness=1)
        cv2.line(img, tuple(tmp[19]), tuple(tmp[20]), color, thickness=1)

        self.tmp = tmp