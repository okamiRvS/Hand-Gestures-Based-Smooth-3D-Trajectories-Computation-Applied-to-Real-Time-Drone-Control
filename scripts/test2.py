import numpy as np
import pandas as pd
import os
import pdb
import csv

pose = []
orientation = []
dtime = []
speed = []
with open('data.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    line_count = 0
    for row in csv_reader:
        print(f'{", ".join(row)}')
        elem = [float(x) for x in row]

        if line_count < 3:
            pose.append(elem)
        elif line_count < 6:
            orientation.append(elem)
        elif line_count == 6:
            dtime.append(elem)
        elif line_count == 7:
            speed.append(elem)
        
        line_count+=1

        print("\n\n")

pose = np.vstack((pose[0], pose[1], pose[2])).T
pdb.set_trace()