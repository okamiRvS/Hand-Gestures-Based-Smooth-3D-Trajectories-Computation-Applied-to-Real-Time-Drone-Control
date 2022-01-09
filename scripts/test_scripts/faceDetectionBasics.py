import cv2
import mediapipe as mp
import time

mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils # this is useful to draw things
faceDetection = mpFaceDetection.FaceDetection()

pTime = 0
cTime = 0

cv2.namedWindow("preview")
cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)

if cap.isOpened(): # try to get the first frame
    success, img = cap.read()
else:
    success = False

while success:
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceDetection.process(imgRGB)
    
    # if an hand is detected return coordinate position with 
    #print(results.multi_hand_landmarks) 

    if results.detections:
        for id, detection in enumerate(results.detections):
            #print(id, detection)
            #print(detection.score)
            #print(detection.location_data.relative_bounding_box)
            mpDraw.draw_detection(img, detection)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3) # print fps

    cv2.imshow("preview", img)
    success, img = cap.read()
    
    key = cv2.waitKey(20)
    if key == 27: # exit on ESC
        break

cv2.destroyWindow("preview")