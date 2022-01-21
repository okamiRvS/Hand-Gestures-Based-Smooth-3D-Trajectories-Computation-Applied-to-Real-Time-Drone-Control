import time, cv2
from threading import Thread
from djitellopy import tello
import os
import pdb


class recordVideo():

    def __init__(self, me):

        self.VIDEO_DIR_PATH = os.path.join('src', 'video_src')

        self.keepRecording = True
        self.recorder = None
        self.getLastVideoIdx = self.getLastIdx()
        self.frame_read = me.get_frame_read()
        self.height, self.width, _ = self.frame_read.frame.shape


    def getLastIdx(self):
        
        if not os.path.exists(self.VIDEO_DIR_PATH):
            if os.name == 'posix': # if linux system
                os.system(f"mkdir -p {CSV_DIR_PATH}")
            if os.name == 'nt': # if windows system
                os.system(f"mkdir {self.VIDEO_DIR_PATH}")

            return 1

        for root, dirs, files in os.walk(self.VIDEO_DIR_PATH, topdown=True):
            return len(files)+1


    def videoRecorder(self):
        # create a VideoWrite object, recoring to ./video.avi
        
        video = cv2.VideoWriter(f'{self.VIDEO_DIR_PATH}/video{self.getLastVideoIdx}.avi', cv2.VideoWriter_fourcc(*'XVID'), 30, (self.width, self.height))

        while self.keepRecording:
            video.write(self.frame_read.frame)
            time.sleep(1 / 30)

        video.release()
        print(f"Video{self.getLastVideoIdx} saved.")


    def run(self):
        # we need to run the recorder in a seperate thread, otherwise blocking options
        #  would prevent frames from getting added to the video
        self.recorder = Thread(target=self.videoRecorder)
        self.recorder.start()


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