import time, cv2
from threading import Thread
from djitellopy import tello
import os
import pdb


class recordVideo():

    def __init__(self, me, path):

        self.keepRecording = True
        self.frame_read = me.get_frame_read()
        self.height, self.width, _ = self.frame_read.frame.shape
        self.path = path 


    def videoRecorder(self):
        """
        create a VideoWrite object, 
        recording to self.VIDEO_DIR_PATH}/video{self.getLastVideoIdx}.avi
        """
        
        video = cv2.VideoWriter(f'{self.path}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))

        while self.keepRecording:

            try:
                video.write(self.frame_read.frame)
            except Exception as e:
                print(e)

            time.sleep(1 / 30)

        video.release()
        print(f"Video {self.path} saved.")


    def run(self):
        """
        we need to run the recorder in a seperate thread, otherwise blocking options
        would prevent frames from getting added to the video
        """
        
        recorder = Thread(target=self.videoRecorder)
        recorder.start()


    def stop(self):
        self.keepRecording = False


def main():

    me = tello.Tello()
    me.connect()
    print(me.get_battery())
    
    obj = recordVideo(me)
    obj.run()


if __name__ == "__main__":
    
    main()