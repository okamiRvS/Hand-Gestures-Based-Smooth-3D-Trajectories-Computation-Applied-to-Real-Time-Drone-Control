from djitellopy import tello
import keyPressModule as kp
import time
import cv2
import handTrackingModule as htm

kp.init()

me = tello.Tello()
me.connect()
print(me.get_battery())
global img
me.streamon() # to get the stream image

detector = htm.handDetector()

def getKeyboardInput():
    #left-right, foward-back, up-down, yaw veloity
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 30

    if kp.getKey("LEFT"): lr = -speed
    elif kp.getKey("RIGHT"): lr = speed

    if kp.getKey("UP"): fb = speed
    elif kp.getKey("DOWN"): fb = -speed

    if kp.getKey("w"): ud = speed
    elif kp.getKey("s"): ud = -speed

    if kp.getKey("a"): yv = speed
    elif kp.getKey("d"): yv = -speed

    if kp.getKey("e"): me.takeoff(); time.sleep(3) # this allows the drone to takeoff
    if kp.getKey("q"): me.land() # this allows the drone to land

    if kp.getKey('z'):
        cv2.imwrite(f'src/tello_screenshots/{time.time()}.jpg', img)
        time.sleep(0.3)

    return [lr, fb, ud, yv]


BEGIN = 0
START = 1
TRACKING = 2

state = BEGIN

queue = []
lenMaxQueue = 20
indexQueue = 0
isQueueNotMaxLimit = True
mean = 0
tolleranceSTART = 10
tolleranceTRACKING = 60

pTime = 0
cTime = 0

while True:
    vals = getKeyboardInput()
    me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
    time.sleep(0.05)

    img = me.get_frame_read().frame
    #img = cv2.resize(img, (360, 240)) # comment to get bigger frames
    img = detector.findHands(img)
    lmList = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList[0])
        val = lmList[0][1] + lmList[0][2] # I'LL DO MEAN FOR X AND MEAN FOR Y, THIS IS JUST TRY

        # fill all the queue before start the mean
        if indexQueue < lenMaxQueue and isQueueNotMaxLimit:
            queue.append(val)
            cv2.putText(img, f"{0}", (200,40), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
            cv2.putText(img, f"{mean}", (200,100), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
            cv2.putText(img, f"{val}", (200,200), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,255), 3)
            indexQueue+=1
        else:
            isQueueNotMaxLimit = False
            if indexQueue == lenMaxQueue:
                indexQueue = 0

            queue[indexQueue] = val 
            
            if state == BEGIN:
                mean = sum(queue) / lenMaxQueue
                checkStart = int(abs(mean - val))

                if checkStart < tolleranceSTART:
                    cv2.putText(img, f"{checkStart}", (200,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)
                    cv2.putText(img, f"{mean}", (200,100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)
                    cv2.putText(img, f"{val}", (200,200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)
                    cv2.putText(img, f"START", (200,300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,0), 3)
                    state = START
                    startingPoint = (lmList[0][1], lmList[0][2])
                else:
                    cv2.putText(img, f"{checkStart}", (200,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    cv2.putText(img, f"{mean}", (200,100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    cv2.putText(img, f"{val}", (200,200), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    cv2.putText(img, f"BEGIN", (200,300), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
            
            if state == START:
                nlastMovements = 5
                tmpList = []
                if indexQueue < nlastMovements:
                    nElementFromEnd = nlastMovements - indexQueue
                    tmpList = queue[(lenMaxQueue - nElementFromEnd)::]
                    tmpList = tmpList + queue[:indexQueue]
                else:
                    tmpList = queue[indexQueue-nlastMovements:indexQueue]
                mean = sum(tmpList) / nlastMovements
                checkStartTracking = int(abs(mean - val))
                endingPoint = (lmList[0][1], lmList[0][2])

                # draw the begin of the trajectory
                cv2.circle(img, startingPoint, radius=0, color=(0,255,0), thickness=-1)

                if checkStartTracking < tolleranceTRACKING:
                    cv2.putText(img, f"{checkStartTracking}", (200,40), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
                    cv2.putText(img, f"{mean}", (200,100), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
                    cv2.putText(img, f"{val}", (200,200), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
                    cv2.putText(img, f"START", (200,300), cv2.FONT_HERSHEY_PLAIN, 1, (255,0,0), 3)
                    cv2.circle(img, endingPoint, radius=0, color=(0,255,0), thickness=-1)
                    cv2.line(img, startingPoint, endingPoint, (255,255,0), thickness=2)
                else:
                    cv2.putText(img, f"{checkStartTracking}", (200,40), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    cv2.putText(img, f"{mean}", (200,100), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    cv2.putText(img, f"{val}", (200,200), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    cv2.putText(img, f"BEGIN", (200,300), cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 3)
                    state = BEGIN      

            if indexQueue < lenMaxQueue:
                indexQueue+=1

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, f"FPS: {int(fps)}", (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3) # print fps

    cv2.imshow("Image", img)
    cv2.waitKey(1)