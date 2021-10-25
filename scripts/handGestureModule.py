import tensorflow as tf
import pandas as pd
import numpy as np
import os

import cv2
import pdb

class handGestureRecognition():

    lastModel = "1635164294"
    export_path = str.encode(os.path.join("Tensorflow","workspace", "models", "my_hand_gesture_model", lastModel)) # must be in bytes
    SPECIES = ['thumbsup', 'thumbsdown', 'thankyou', 'livelong']

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

        '''
        for idx, resultPred in enumerate(predictions["class_ids"]):
            class_id = resultPred[0]
            probability = predictions['probabilities'][idx][class_id]
            print(probability)
            print(f"\tPrediction is {self.SPECIES[class_id]} {100 * probability :.2f}%")
            print(f"\tExpected: {self.SPECIES[test_y.iloc[idx]]}")
        '''
        

        return predictions

    def drawHandGesture(self):
        # Each location contains positions in order: top, right, bottom, left
        top_left = (face_location[3], face_location[0])
        bottom_right = (face_location[1], face_location[2])

        # Get color by name using our fancy function
        color = self.name_to_color(match)

        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, self.FRAME_THICKNESS)

        # Now we need smaller, filled grame below for a name
        # This time we use bottom in both corners - to start from bottom and move 50 pixels down
        top_left = (face_location[3], face_location[2])
        bottom_right = (face_location[1], face_location[2] + 22)

        # Paint frame
        cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

        # Wite a name
        cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.FONT_THICKNESS)

    def processFaces(self, image):
        #image = self.image_resize(image, height = 800)

        # This time we first grab face locations - we'll need them to draw boxes
        locations = face_recognition.face_locations(image, model=self.MODEL)

        # Now since we know loctions, we can pass them to face_encodings as second argument
        # Without that it will search for faces once again slowing down whole process
        encodings = face_recognition.face_encodings(image, locations)

        # We passed our image through face_locations and face_encodings, so we can modify it
        # First we need to convert it from RGB to BGR as we are going to work with cv2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # But this time we assume that there might be more faces in an image - we can find faces of different people
        print(f', found {len(encodings)} face(s)')
        for face_encoding, face_location in zip(encodings, locations):

            # We use compare_faces (but might use face_distance as well)
            # Returns array of True/False values in order of passed known_faces
            results = face_recognition.compare_faces(self.known_faces, face_encoding, self.TOLERANCE)

            # Since order is being preserved, we check if any face was found then grab index
            # then label (name) of first matching known face withing a tolerance
            match = None
            if True in results:  # If at least one is true, get a name of first of found labels
                match = self.known_names[results.index(True)]
                print(f' - {match} from {results}')

                # Each location contains positions in order: top, right, bottom, left
                top_left = (face_location[3], face_location[0])
                bottom_right = (face_location[1], face_location[2])

                # Get color by name using our fancy function
                color = self.name_to_color(match)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, self.FRAME_THICKNESS)

                # Now we need smaller, filled grame below for a name
                # This time we use bottom in both corners - to start from bottom and move 50 pixels down
                top_left = (face_location[3], face_location[2])
                bottom_right = (face_location[1], face_location[2] + 22)

                # Paint frame
                cv2.rectangle(image, top_left, bottom_right, color, cv2.FILLED)

                # Wite a name
                cv2.putText(image, match, (face_location[3] + 10, face_location[2] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), self.FONT_THICKNESS)

        return image


def main():
    print("hello")

if __name__ == "__main__":
    main()