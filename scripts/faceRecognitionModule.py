import face_recognition
import os
import cv2
import pdb

class faceRecognition():
    KNOWN_FACES_DIR = 'src/known_faces'
    UNKNOWN_FACES_DIR = 'unknown_faces'
    TOLERANCE = 0.6
    FRAME_THICKNESS = 3
    FONT_THICKNESS = 2
    MODEL = 'hog'  # default: 'hog', other one can be 'cnn' - CUDA accelerated (if available) deep-learning pretrained model

    def __init__(self):        
        print('Loading known faces...')
        self.known_faces = []
        self.known_names = []

        # We oranize known faces as subfolders of KNOWN_FACES_DIR
        # Each subfolder's name becomes our label (name)
        for name in os.listdir(self.KNOWN_FACES_DIR):

            # Next we load every file of faces of known person
            for filename in os.listdir(f'{self.KNOWN_FACES_DIR}/{name}'):

                # Load an image
                image = face_recognition.load_image_file(f'{self.KNOWN_FACES_DIR}/{name}/{filename}')

                # Get 128-dimension face encoding
                # Always returns a list of found faces, for this purpose we take first face only (assuming one face per image as you can't be twice on one image)
                try:
                    encoding = face_recognition.face_encodings(image)[0]
                    # Append encodings and name
                    self.known_faces.append(encoding)
                    self.known_names.append(name)
                except:
                    print("this name: " + filename + " didn't work because pheraps no face detected")

    # Returns (R, G, B) from name
    def name_to_color(self, name):
        # Take 3 first letters, tolower()
        # lowercased character ord() value rage is 97 to 122, substract 97, multiply by 8
        color = [(ord(c.lower())-97)*8 for c in name[:3]]
        return color

    def image_resize(self, image, width = None, height = None, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized

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

'''
# Show image
cv2.namedWindow(filename, cv2.WINDOW_AUTOSIZE) 
cv2.moveWindow(filename, 0, 0) 
cv2.imshow(filename, image)
k = cv2.waitKey(0)
if k==27:
    break
cv2.destroyWindow(filename)
'''