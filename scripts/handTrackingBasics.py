import cv2
import mediapipe as mp
import time

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils # this is useful to draw things

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
    results = hands.process(imgRGB)
    
    # if an hand is detected return coordinate position with 
    #print(results.multi_hand_landmarks) 

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                print(id, cx, cy) # here we know the pixel of the dots-hand
                if id == 4: #here we draw a specific dot
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            
            #mpDraw.draw_landmarks(img, handLms) # we draw each hand detected
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS) # we draw each hand detected

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