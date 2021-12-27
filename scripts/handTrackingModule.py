import cv2
import mediapipe as mp
import time
import pdb


class handDetector():

    def __init__(self, mode=False, maxHands = 2, detectionCon = 0.5, trackCon=0.5 ):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        # we need to initialize them
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, 
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils # this is useful to draw things


    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if draw:
            self.results = self.hands.process(imgRGB)

            # if an hand is detected return coordinate position with 
            #print(results.multi_hand_landmarks) 
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    #mpDraw.draw_landmarks(img, handLms) # we draw each hand detected
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS) # we draw each hand detected
        
        return img


    def findPosition(self, img, handNo=0, draw=True):

        #this is to find a position for a specific hand.

        lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            #print(self.results.multi_handedness[0].classification[0].label)
            
            for id, lm in enumerate(myHand.landmark):


                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                # here we know the pixel of the dots-hand
                lmList.append([id, cx, cy])

                if draw:
                    
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

                    # print depth
                    if id == 0 or id == 12 or id == 4 or id == 20:
                        img = cv2.putText(img, f'{round(lm.z, 3)}', (cx+2, cy+2), cv2.FONT_HERSHEY_SIMPLEX, 
                                        0.6, (0, 255, 0), 1, cv2.LINE_AA)


        return lmList


def main():

    pTime = 0
    cTime = 0

    detector = handDetector()

    cv2.namedWindow("preview")
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    if cap.isOpened(): # try to get the first frame
        success, img = cap.read()
        detector.findHands(img)
    else:
        success = False

    while success:
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # print fps
        cv2.imshow("preview", img)

        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.findHands(img)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[0])
        
        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break

    cv2.destroyWindow("preview")

if __name__ == "__main__":
    
    main()