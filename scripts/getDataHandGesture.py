from djitellopy import tello
import time
import cv2
import handTrackingModule as htm
import pandas as pd 
import numpy as np
import pdb

me = tello.Tello()
me.connect()
print(me.get_battery())
global img

detector = htm.handDetector()

# 2. Define Images to Collect
labels = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']
number_imgs = 10

np_array = np.zeros((len(labels)*number_imgs, 21*2 + 1), dtype=np.int32)

me.streamon() # to get the stream image

def getReadyForTheNextAction(action):
    #This will run for 15 s
    t_end = time.time() + 15.0
    while time.time() < t_end:
        img = me.get_frame_read().frame
        cv2.imshow("Image", img)
        secondLeft = int(t_end-time.time())
        cv2.putText(img, f"PHOTO about {label} TAKE in: {secondLeft}s", (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3) # print fps
        cv2.waitKey(1)

for idx, label in enumerate(labels):
    print('Collecting images for {}'.format(label))
    getReadyForTheNextAction(label)

    for imgnum in range(number_imgs):
        isLmListEmpty = True
        while isLmListEmpty:
            print('Collecting image {}'.format(imgnum))
            time.sleep(3)

            img = me.get_frame_read().frame
            #img = cv2.resize(img, (360, 240)) # comment to get bigger frames

            img = detector.findHands(img)
            lmList = detector.findPosition(img, draw=False)
            
            if len(lmList) != 0:
                index = idx*number_imgs + imgnum

                # insert values, except for the last element for each row because it is the label that we set after this iteration
                for j, val in enumerate(lmList[:-1]):
                    np_array[index, j*2] = val[1]
                    np_array[index, j*2+1] = val[2]

                np_array[index, -1] = idx

                isLmListEmpty = False


                print(lmList)
                print(f"battery is: {me.get_battery()}")
                print("\n\n")

            
            cv2.putText(img, f"Collecting images for: {label}", (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)
            cv2.putText(img, f"photo number: {imgnum+1}/{number_imgs}", (30,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

            cv2.imshow("Image", img)
            cv2.waitKey(1)


pd.DataFrame(np_array).to_csv("dataHandGesture.csv", index=False, header=None)
me.streamoff() # to close stream
cv2.destroyAllWindows()