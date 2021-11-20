from djitellopy import tello
import time
import cv2
import handTrackingModule as htm
import pandas as pd 
import numpy as np
import os
import json
import pdb

global img
getFromWebcam = True

# if true you can save all picture of the data and decide the currentPicture, if false you'll use that data that can
# be uncorrect (could be useful for real application?)
takeControl = False

# wait from a frame and another
timeDelay = 0

# define images to collect
labels = ['stop', 'onefingerup', 'twofingerup', 'thumbsup'] # da migliorare con movimenti avanti e indietro
number_imgs = 621

# (207*3)
# IT'S REALLY IMPORTANT DIVERSIFY THE ORIENTATION THROUGH ROLL, PITCH AND YAW
# 0-206 z1 (background)
    # (23*3)
    # back
        # right 0-22
        # center 23-45
        # left 46-68
    # center
        # right 68-91
        # center 92-114
        # left 115-137
    # front
        # right 138-160
        # center 161-183
        # left 184-206

# 207-413 z2 (center)
    # back
        # right 207-229
        # center 230-252
        # left 253-275
    # center
        # right 276-298
        # center 299-321
        # left 322-344
    # front
        # right 345-367
        # center 368-390
        # left 391-413

# 414-620 z3 (foreground)
    # back
        # right 414-436
        # center 437-459
        # left 460-482
    # center
        # right 483-505
        # center 506-528
        # left 529-551
    # front
        # right 552-574
        # center 575-597
        # left 598-620

np_array = np.zeros((len(labels)*number_imgs, 21*2 + 1), dtype=np.int32)

detector = htm.handDetector()

# HERE MAYBE COULD BE USEFUL USE A FACTORY FUNCTION (FROM SOFTWARE ENGENEERING)
if getFromWebcam:
    # OPEN WEBCAM
    cv2.namedWindow("Image")
    cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if cap.isOpened(): # try to get the first frame
        success, img = cap.read()
    else:
        success = False
else:
    me = tello.Tello()
    me.connect()
    print(me.get_battery())
    me.streamon() # to get the stream image

def getReadyForTheNextAction(action):
    #This will run for 15 s
    t_end = time.time() + 10.0
    while time.time() < t_end:

        if getFromWebcam:
            success, img = cap.read()
        else:
            img = me.get_frame_read().frame
        
        img = cv2.flip(img, 1)

        fontScale = 1
        font = cv2.FONT_HERSHEY_DUPLEX
        thickness = 2

        secondLeft = int(t_end-time.time())
        cv2.putText(img, f"\"{action}\" in: {secondLeft}s", (10,40), font, fontScale, (0,255,0), thickness)

        if takeControl:
            cv2.putText(img, f"click ESC to delete the picture", (10,70), font, fontScale, (0,255,0), thickness)
            cv2.putText(img, f"click BACKSPACE to continue", (10,100), font, fontScale, (0,255,0), thickness)

        cv2.imshow("Image", img)
        cv2.waitKey(1)

CSV_DIR_PATH = os.path.join('src', 'dataHandGesture')
if not os.path.exists(CSV_DIR_PATH):
    if os.name == 'posix': # if linux system
        os.system(f"mkdir -p {CSV_DIR_PATH}")
    if os.name == 'nt': # if windows system
        os.system(f"mkdir {CSV_DIR_PATH}")

STATE_PATH = os.path.join(CSV_DIR_PATH, 'state.json')

nAttempt = 0
nLabel = 0
nImg = 0

CSV_PATH = os.path.join(CSV_DIR_PATH, f"file_{nAttempt}.csv")

if os.path.exists(STATE_PATH):
    # read json state file
    with open(STATE_PATH) as json_file:
        data = json.load(json_file)

    nAttempt = data["nAttempt"]
    nLabel = data["nLabel"]
    nImg = data["nImg"]

    CSV_PATH = os.path.join(CSV_DIR_PATH, f"file_{nAttempt}.csv")
    
    if os.path.exists(CSV_PATH):
        # restore data
        df = pd.read_csv(CSV_PATH, sep=',',header=None)
        np_array = df.to_numpy()
        
        # update values for storing in a new csv file
        nAttempt+=1
        CSV_PATH = os.path.join(CSV_DIR_PATH, f"file_{nAttempt}.csv")
    else:
        raise ValueError('You have the state.json without the CSV.')
else:
    if os.path.exists(CSV_PATH):
        raise ValueError('You have the CSV without the state.json')

    # create json state file
    with open(STATE_PATH, 'w', encoding='utf-8') as f:
        data = {
            "nAttempt": nAttempt,
            "nLabel": nLabel,
            "nImg": nImg,
        }
        json.dump(data, f, ensure_ascii=False, indent=4)

# MAYBE THIS PART IT'S NOT REALLY USEFUL, MAYBE IT WORKS JUST WITH THE FOLLOW CODE TO CREATE FOLDER FOR EACH LABEL
# create if not exist label folders in imgData 
'''
IMGDATA_DIR_PATH = os.path.join('src', 'dataHandGesture', 'imgData')
if not os.path.exists(IMGDATA_DIR_PATH):
    if os.name == 'posix': # if linux system
        os.system(f"mkdir -p {IMGDATA_DIR_PATH}")
    if os.name == 'nt': # if windows system
        os.system(f"mkdir {IMGDATA_DIR_PATH}")
'''

try:
    for label in labels[nLabel:]:

        if takeControl:
            # create label folder
            LABEL_DIR_PATH = os.path.join('src', 'dataHandGesture', 'imgData', label)
            if not os.path.exists(LABEL_DIR_PATH):
                if os.name == 'posix': # if linux system
                    os.system(f"mkdir -p {LABEL_DIR_PATH}")
                if os.name == 'nt': # if windows system
                    os.system(f"mkdir {LABEL_DIR_PATH}")

        print(f"Collecting images for {label}")
        getReadyForTheNextAction(label)

        for imgnum in range(number_imgs):

            if imgnum >= nImg: # to restore at the last img

                isLmListEmpty = True
                while isLmListEmpty:

                    print(f"Collecting image {imgnum}")
                    time.sleep(timeDelay)

                    if getFromWebcam:
                        success, img = cap.read()
                    else:
                        img = me.get_frame_read().frame
                    
                    #img = cv2.resize(img, (360, 240)) # comment to get bigger frames
                    img = cv2.flip(img, 1) # this is important

                    img = detector.findHands(img)
                    lmList = detector.findPosition(img, draw=False)
                    
                    if len(lmList) != 0:

                        index = nLabel*number_imgs + imgnum # current_row

                        x_sum = y_sum = 0
                        # insert values, except for the last element for each row because it is the label that we set after this iteration
                        for j, val in enumerate(lmList):
                            np_array[index, j*2] = val[1]
                            x_sum += val[1]
                            np_array[index, j*2+1] = val[2]
                            y_sum += val[2]

                        x_mean = x_sum / 21
                        y_mean = y_sum / 21

                        # translate all values to the origin
                        for j in range(42):
                            if j%2 == 0:
                                np_array[index, j] = np_array[index, j] - x_mean
                            else:
                                np_array[index, j] = np_array[index, j] - y_mean

                        np_array[index, -1] = nLabel # put label on last column

                        isLmListEmpty = False

                        #print(lmList)
                        print(np_array[index])

                        if not getFromWebcam:
                            print(f"battery is: {me.get_battery()}")
                            print("\n\n")

                    
                    cv2.putText(img, f"Collecting images for: {label}", (10,40), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)
                    cv2.putText(img, f"photo number: {imgnum+1}/{number_imgs}", (30,100), cv2.FONT_HERSHEY_PLAIN, 2, (255,0,255), 3)

                    cv2.imshow("Image", img)
                    key = cv2.waitKey(0)

                    if takeControl:
                        if key == 32: # backspace
                            cv2.imwrite(os.path.join(LABEL_DIR_PATH, f"{imgnum}.jpeg"), img)
                        elif key == 27: # exit
                            isLmListEmpty = True # this means that the picture wans't good, so skip that

                if nImg+1 < number_imgs:
                    nImg+=1

        if nLabel+1 < len(labels):
            nLabel+=1
            nImg = 0

except:
    print("An error occurred")


# save the current state
with open(STATE_PATH, 'w', encoding='utf-8') as f:
    data = {
        "nAttempt": nAttempt,
        "nLabel": nLabel,
        "nImg": nImg,
    }
    json.dump(data, f, ensure_ascii=False, indent=4)

pd.DataFrame(np_array).to_csv(CSV_PATH, index=False, header=None)
if not getFromWebcam:
    me.streamoff() # to close stream
cv2.destroyAllWindows()