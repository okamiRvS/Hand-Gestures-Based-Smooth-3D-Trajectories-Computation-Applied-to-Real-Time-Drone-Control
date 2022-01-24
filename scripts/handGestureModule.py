import tensorflow as tf
import pandas as pd
import numpy as np
import os

import cv2
import pdb


class handGestureRecognition():

    # To train a new model look at "3. Hand gesture recognition.ipynb" use as kernel tfod

    lastModel = "1642981324" # COPY THE FOLDER NAME OF Tensorflow/workspace/models/my_hand_gesture_model
    
    if os.name == 'posix': # if linux system
        export_path = str.encode(os.path.join("/home/usiusi/catkin_ws/src/DJI-Tello-3D-Hand-Gesture-control/Tensorflow/workspace/models/my_hand_gesture_model", lastModel)) # must be in bytes
    elif os.name == 'nt': # if windows system
        export_path = str.encode(os.path.join("Tensorflow","workspace", "models", "my_hand_gesture_model", lastModel)) # must be in bytes

    #SPECIES = ['stop', 'onefingerup', 'twofingerup', 'thumbsup']
    SPECIES = ['backward', 'detect', 'down', 'forward', 'land', 'left', 'ok', 'right', 'stop', 'up']
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2

    def __init__(self):

        # Loading the estimator
        self.predict_fn = tf.saved_model.load(self.export_path).signatures['predict']


    def processHands(self, img, handPoints):

        # np_array = np.zeros((1, 21*2), dtype=np.int32)

        # x_sum = y_sum = 0
        # for j, val in enumerate(handPoints):
        #     np_array[0, j*2] = val[1]
        #     x_sum += val[1]
        #     np_array[0, j*2+1] = val[2]
        #     y_sum += val[2]

        # x_mean = x_sum / 21
        # y_mean = y_sum / 21

        # # translate all values to the origin
        # for j in range(42):
        #     if j%2 == 0:
        #         np_array[0, j] = np_array[0, j] - x_mean
        #     else:
        #         np_array[0, j] = np_array[0, j] - y_mean

        np_array = handPoints.getPointsForNet()

        CSV_COLUMN_NAMES = np.arange(42)
        CSV_COLUMN_NAMES = [str(item) for item in CSV_COLUMN_NAMES]

        self.POINTS = pd.DataFrame([np_array], columns=CSV_COLUMN_NAMES)
        predictions = self.getPredictions()      

        for idx, resultPred in enumerate(predictions["class_ids"]):
            class_id = resultPred[0]
            probability = predictions['probabilities'][idx][class_id]
            outputClass = self.SPECIES[class_id]

            #print(f"\tPrediction is {outputClass} {100 * probability :.2f}%")

        self.drawHandGesture(img, handPoints, outputClass, probability)

        return img, outputClass, probability


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


    def drawHandGesture(self, img, handPoints, match, prob):

        handPoints = handPoints.lmList

        arr = np.matrix(handPoints)
        max_val = arr.max(0)
        min_val = arr.min(0)

        # Each location contains positions in order: top, right, bottom, left
        top_left = (min_val[0,1], max_val[0,2])
        bottom_right = (max_val[0,1], min_val[0,2])

        # Get color by name using a fancy function
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

        # Write probability
        cv2.putText(img, f"{100 * prob :.2f}%", (min_val[0,1] + 100, min_val[0,2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.FONT_THICKNESS)


def main():

    print("hello")


if __name__ == "__main__":
    
    main()