from djitellopy import tello
from time import time
import cv2
import numpy as np
import pandas as pd
import os
import json

'''
lo = [1,3,60,34,12,11]
tmp = lo[:2]
print(tmp)
print(lo[-2:])
tmp = tmp + lo[-2::]
print(tmp)
'''

'''
lo = [1,3,60,34,12,11]
for j, val in enumerate(lo[:-1]):
    print(val)

lo[-1] = 34
print(lo)
'''

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

np_array = np.zeros((3,5), dtype=np.int32)

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
    
print(np_array)

try:
    np_array[1,2] = 1
    raise ValueError('An error occurred.') # just for test
except:
    print("An error occurred")

with open(STATE_PATH, 'w', encoding='utf-8') as f:
    data = {
        "nAttempt": nAttempt,
        "nLabel": nLabel,
        "nImg": nImg,
    }
    json.dump(data, f, ensure_ascii=False, indent=4)


pd.DataFrame(np_array).to_csv(CSV_PATH, index=False, header=None)