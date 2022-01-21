import time, cv2
from threading import Thread
from djitellopy import Tello

keepRecording = True
frame_read = tello.get_frame_read()

class recordVideo():

    def __init__(self, me):
        self.frame_read = me
        self.keepRecording = True
        self.recorder = None

    def videoRecorder(self):
        # create a VideoWrite object, recoring to ./video.avi
        height, width, _ = self.frame_read.frame.shape
        video = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (width, height))

        while self.keepRecording:
            video.write(self.frame_read.frame)
            time.sleep(1 / 30)

        video.release()


    def run(self):
        # we need to run the recorder in a seperate thread, otherwise blocking options
        #  would prevent frames from getting added to the video
        self.recorder = Thread(target=self.videoRecorder)
        self.recorder.start()

    def stop(self):
        self.keepRecording = False
        self.recorder.join()


def main():

    obj = recordVideo()
    obj.run()


if __name__ == "__main__":
    
    main()