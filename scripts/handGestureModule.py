import tensorflow as tf
import pandas as pd
import numpy as np
import os

import cv2
import pdb

class handGestureRecognition():

    lastModel = "1636127582" # COPY THE FOLDER NAME OF Tensorflow/workspace/models/my_hand_gesture_model
    export_path = str.encode(os.path.join("Tensorflow","workspace", "models", "my_hand_gesture_model", lastModel)) # must be in bytes
    SPECIES = ['stop', 'indexup', 'twofingerup', 'punch', 'thumbsup']
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2

    def __init__(self):        
        # Loading the estimator
        self.predict_fn = tf.saved_model.load(self.export_path).signatures['predict']

    def processHands(self, img, handPoints):
        np_array = np.zeros((1, 21*2), dtype=np.int32)

        for j, val in enumerate(handPoints):
            np_array[0, j*2] = val[1]
            np_array[0, j*2+1] = val[2]


        CSV_COLUMN_NAMES = np.arange(42)
        CSV_COLUMN_NAMES = [str(item) for item in CSV_COLUMN_NAMES]

        self.POINTS = pd.DataFrame(np_array, columns=CSV_COLUMN_NAMES)
        predictions = self.getPredictions()

        for idx, resultPred in enumerate(predictions["class_ids"]):
            class_id = resultPred[0]
            probability = predictions['probabilities'][idx][class_id]
            print(f"\tPrediction is {self.SPECIES[class_id]} {100 * probability :.2f}%")

        self.drawHandGesture(img, handPoints, self.SPECIES[class_id])

        return img

    def getPredictions(self):
        # Convert input data into serialized Example strings.

        examples = []
        for index, row in self.POINTS.iterrows():
            feature = {}
            for col, value in row.iteritems():
                feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature
                )
            )
            examples.append(example.SerializeToString())

        # Convert from list to tensor
        examples = tf.constant(examples)

        # make predictions of all testset
        predictions = self.predict_fn(examples=examples)

        return predictions

    # Returns (R, G, B) from name
    def name_to_color(self, name):
        # Take 3 first letters, tolower()
        # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        color = [(ord(c.lower())-97)*8 for c in name[:3]]
        return color

    def drawHandGesture(self, img, handPoints, match):
        arr = np.matrix(handPoints)
        max_val = arr.max(0)
        min_val = arr.min(0)

        # Each location contains positions in order: top, right, bottom, left
        top_left = (min_val[0,1], max_val[0,2])
        bottom_right = (max_val[0,1], min_val[0,2])

        # Get color by name using our fancy function
        color = self.name_to_color(match)

        # Paint frame
        cv2.rectangle(img, top_left, bottom_right, color, self.FRAME_THICKNESS)

        # Now we need smaller, filled grame below for a name
        # This time we use bottom in both corners - to start from bottom and move 50 pixels down
        top_left = (min_val[0,1], min_val[0,2])
        bottom_right = (max_val[0,1], min_val[0,2] + 22)

        # Paint frame
        cv2.rectangle(img, top_left, bottom_right, color, cv2.FILLED)

        # Wite a name
        cv2.putText(img, match, (min_val[0,1] + 10, min_val[0,2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.FONT_THICKNESS)

def main():
    print("hello")

if __name__ == "__main__":
    main()