import cv2
import mediapipe as mp
import time
import handTrackingModule as htm
import faceRecognitionModule as frm

pTime = 0
cTime = 0

detector = htm.handDetector()
face = frm.faceRecognition()

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

# OPEN WEBCAM
if cap.isOpened(): # try to get the first frame
    success, img = cap.read()
    img = detector.findHands(img)
    img = face.processFaces(img)
else:
    success = False

while success:
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3) # print fps
    cv2.imshow("preview", img)

    success, img = cap.read()
    img = detector.findHands(img)
    #img = face.processFaces(img)
    lmList = detector.findPosition(img, draw=False)
    if len(lmList) != 0:
        print(lmList[0])
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")