from djitellopy import tello
import keyPressModule as kp
import recordVideoModule as recVid
import time
import cv2
import numpy as np
import math
import pdb
import fullControllModule
import matplotlib.pyplot as plt
import os
import pandas as pd


class keyboardControl:

    def __init__(self):

        self.VIDEO_DIR_PATH = os.path.join('src', 'video_src')

        ######################################################
        # PARAMETERS
        fSpeed = 117/10 # forward speed in cm/2     (15cm/s)
        aSpeed = 360/10 # angular speed degrees/s   (50d/s)
        self.interval = 0.25

        self.dInterval = fSpeed*self.interval
        self.aInterval = aSpeed*self.interval

        ######################################################
        self.x, self.y, self.z = 500, 500, 500
        self.a = 0
        self.yaw = 0
        height = 0
        self.totTime = 0

        self.points = [(500, 500, 500)]

        self.flag = True       


    def setLastIdx(self) -> int:
        """
        Create folder "self.VIDEO_DIR_PATH" if doesn't exist and return 1.
        In general, return as index the number of video added +1.

        If folder with n as name already exists, try to create folder
        with n+1 as name, otherwise iterate.

        Update self.VIDEO_DIR_PATH with folder where to put all files
        """
        
        if not os.path.exists(self.VIDEO_DIR_PATH):
            if os.name == 'posix': # if linux system
                os.system(f"mkdir -p {self.VIDEO_DIR_PATH}")
                os.system(f"mkdir -p {self.VIDEO_DIR_PATH}\\1")
            if os.name == 'nt': # if windows system
                os.system(f"mkdir {self.VIDEO_DIR_PATH}")
                os.system(f"mkdir {self.VIDEO_DIR_PATH}\\1")

            self.VIDEO_DIR_PATH = os.path.join(self.VIDEO_DIR_PATH, str(1))
            return 1

        nu = len(next(os.walk(self.VIDEO_DIR_PATH))[1]) + 1
        while True:
            # Count number of folders
            folder = os.path.join(self.VIDEO_DIR_PATH, str(nu))
            if not os.path.exists(folder):
                if os.name == 'posix': # if linux system
                    os.system(f"mkdir -p {folder}")
                if os.name == 'nt': # if windows system
                    os.system(f"mkdir {folder}")
                
                self.VIDEO_DIR_PATH = folder
                return nu
            else:
                nu+=1


    def getKeyboardInput2(self, vels):
        const = 117/10 # forward speed in cm/2     (15cm/s)
        dInterval = const*self.interval

        lr, fb, ud, yv = 0, 0, 0, 0

        for i, vel in enumerate(vels):
            # if first index of vels
            if self.totTime < vel[3]:

                lr, ud, fb, _, _ = vel

                lr_speed = (lr/15) * const * self.interval
                lr_interval = int(lr_speed)

                ud_speed = (ud/15) * const * self.interval
                ud_interval = int(ud_speed)

                fb_speed = (fb/15) * const * self.interval
                fb_interval = int(fb_speed)

                break

        # If end trajectory then set self.flag to false for land and stop rec
        if self.totTime > vels[-1][3] + 2 and self.flag:
            self.flag = False

        time.sleep(self.interval)
        self.totTime += self.interval

        lr = int(-lr)
        fb = int(fb)
        ud = int(ud)

        if lr !=0 and fb!=0 and ud !=0:
            self.x += -lr_interval
            self.y += fb_interval
            self.z -= ud_interval
        
        #print(lr, fb, ud, yv, self.x, self.y, self.z)

        return [lr, fb, ud, yv, self.x, self.y, self.z]
        

    def getKeyboardInput(self, me) -> list:
        """
        Get keyboard input to move the drone a fixed speed using a controller.
        Speed can be increased or decresed dinamically.

        Arguments:
            me: this permits to takeoff or land the drone
            img: save this img if getKey('z')
        """

        #left-right, foward-back, up-down, yaw veloity
        lr, fb, ud, yv = 0, 0, 0, 0
        speed = 15
        aSpeed = 50
        d = 0 # distance will be reset each time
        height = 0 #height will be reset each time

        if kp.getKey("LEFT"): 
            lr = -speed
            d = self.dInterval
            a = -180

        elif kp.getKey("RIGHT"): 
            lr = speed
            d = -self.dInterval
            a = 180

        if kp.getKey("UP"):  # this is forward, not up...
            fb = speed
            d = self.dInterval
            a = 270

        elif kp.getKey("DOWN"): # this is backward, not down...
            fb = -speed
            d = -self.dInterval
            a = -90

        if kp.getKey("w"): 
            ud = speed
            height = -self.dInterval

        elif kp.getKey("s"): 
            ud = -speed
            height = self.dInterval

        if kp.getKey("a"): 
            yv = -aSpeed
            self.yaw -= self.aInterval

        elif kp.getKey("d"): 
            yv = aSpeed
            self.yaw += self.aInterval

        if kp.getKey("e"): me.takeoff(); time.sleep(3) # this allows the drone to takeoff
        if kp.getKey("q"): me.land() # this allows the drone to land

        if kp.getKey('z'):
            img = me.get_frame_read().frame
            img = cv2.flip(img, 1)
            cv2.imwrite(f'src/tello_screenshots/{time.time()}.jpg', img)
            time.sleep(0.3)
            return [lr, fb, ud, yv, self.x, self.y, self.z]

        time.sleep(self.interval)
        self.totTime += self.interval

        self.a += self.yaw
        self.x += int(d*math.cos(math.radians(a)))
        self.y += int(d*math.sin(math.radians(a)))
        self.z += int(height)

        return [lr, fb, ud, yv, self.x, self.y, self.z]


    def drawXYPoints(self, img):
        for point in self.points:
            cv2.circle(img, (point[0], point[1]), 5, (0,0,255), cv2.FILLED)
        
        # print last point in green
        cv2.circle(img, (self.points[-1][0], self.points[-1][1]), 8, (0,255,0), cv2.FILLED)

        # print coordinate of the last position
        cv2.putText(img,
                    f"({ (self.points[-1][0] - 500) / 100}, { (self.points[-1][1] - 500) /100}, { (self.points[-1][2] - 500) /100})m",
                    ( self.points[-1][0]+10, self.points[-1][1]+30 ),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,0,255),
                    1) # this give us position in meters not in cm

        # print totTime
        cv2.putText(img, f"{self.totTime}s",
                ( 10, 30 ),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0,255,0),
                1) # this give us position in meters not in cm

    def drawXZPoints(self, img):
        for point in self.points:
            cv2.circle(img, (point[0], point[2]), 5, (0,0,255), cv2.FILLED)
        
        # print last point in green
        cv2.circle(img, (self.points[-1][0], self.points[-1][2]), 8, (0,255,0), cv2.FILLED)

        # print coordinate of the last position
        cv2.putText(img,
                    f"({ (self.points[-1][0] - 500) / 100}, { (self.points[-1][1] - 500) /100}, { (self.points[-1][2] - 500) /100})m",
                    ( self.points[-1][0]+10, self.points[-1][2]+30 ),
                    cv2.FONT_HERSHEY_PLAIN,
                    1,
                    (255,0,255),
                    1) # this give us position in meters not in cm

        # print totTime
        cv2.putText(img, f"{self.totTime}s",
                ( 10, 30 ),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0,255,0),
                1) # this give us position in meters not in cm


    def normalizeData(self, resTraj, height, width, path, log=True):

        #y = np.zeros_like(resTraj[1])

        # traslate everything to the origin, remember that x and y (so resTraj[0] and
        # resTraj[2]) are values between 0 and 1
        xz = np.concatenate(([resTraj[0]], [resTraj[2]]), axis=0)
        xz[0] = xz[0] - xz[0][0]
        xz[1] = xz[1] - xz[1][0]
        y = resTraj[1] - resTraj[1][0] ###################################
        
        # Scale x,z coordinates wrt max value (is just one, the biggest)
        maxXZ = np.max(np.abs(xz))
        xz = xz / maxXZ

        # Remember that width could be different from height, therefore 
        # we need to scale width axis wrt aspect ratio
        aspectRatio = width/height
        xz[1] = xz[1] / aspectRatio 

        fig = plt.figure() 
        plt.subplot(2, 2, 1)
        plt.title('XY')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.xlim(-1, 1)
        plt.ylim(-1, 1) 
        plt.plot(xz[0], y)

        plt.subplot(2, 2, 2)
        plt.title('XZ')
        plt.xlabel('X')
        plt.ylabel('Z')
        plt.xlim(-1, 1) 
        plt.ylim(-1/aspectRatio, 1/aspectRatio) 
        plt.plot(xz[0], xz[1])

        # move coordinates of traj in range from -range cm to range cm
        range = 150
        xz = xz * range
        y = y * (range/2) # reduce range of z space... ########################################

        plt.subplot(2, 2, 3)
        plt.title('XZ')
        plt.xlabel('X cm')
        plt.ylabel('Y cm')
        plt.xlim(-range, range)
        plt.ylim(-range, range) 
        plt.plot(xz[0], y)

        plt.subplot(2, 2, 4)
        plt.title('XZ')
        plt.xlabel('X cm')
        plt.ylabel('Z cm')
        plt.xlim(-range, range)
        plt.ylim(-range, range) 
        plt.plot(xz[0], xz[1])

        plt.savefig(f'{path}_normalized.png')

        if log:
            plt.show()

        # delta space in cm  
        xyz = np.concatenate((xz, [y]), axis=0)
        dspace = np.diff(xyz)

        # scale time from 0 to 10
        istTime = resTraj[6] - resTraj[6][0]
        istTime = istTime / np.max(istTime) * 10
        #istTime = resTraj[6]

        # delta time in secs
        dtime = np.diff(istTime)

        # vels (x,y,z) in cm / s
        vels = dspace / dtime

        # we add time as last coordinate
        vels = np.concatenate( (vels, [istTime[1:]], [dtime] )).T

        # Export normalized data as CSV
        pd.DataFrame(vels).to_csv(f"{path}_normalized.csv", index=False, header=None)

        return vels


    def runJustDrone(self):
        """
        IS ALWAYS isWebcam=False, DRONE FLY, VIDEO RECORDED FROM DRONE,
        DETECTION TRAJECTORY FROM DRONE
        """

        self.isWebcam = False

        me = tello.Tello()
        me.connect()
        print(me.get_battery())

        # Path for save things
        idx = self.setLastIdx()
        path = f"{self.VIDEO_DIR_PATH}\\{idx}"

        fullControll = fullControllModule.FullControll()

        # Reset values
        fullControll.autoSet(path, isWebcam=self.isWebcam, resize=False, showPlot=False)

        # Get the stream image
        me.streamon() # to get the stream image
        time.sleep(3)
       
        # Start rec video
        rec = recVid.recordVideo(me, f"{path}_droneCamera")
        rec.run()

        # Takeoff
        me.takeoff()
        time.sleep(3) 

        # Fly up a bit
        me.send_rc_control(0, 0, 20, 0)
 
        # Get data from hand
        resTraj = fullControll.run(me)

        # Export original data as CSV
        pd.DataFrame(np.array(resTraj)).to_csv(f"{path}_original.csv", index=False, header=None)

        # Get resolution
        height, width = fullControll.getResolution()

        velocities = self.normalizeData(resTraj, height, width, log=False)

        while self.flag:

            # Control with detected trajectory
            vals = self.getKeyboardInput2(velocities)

            imgXY = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
            imgXZ = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
            
            if self.points[-1][0] != vals[4] or self.points[-1][1] != vals[5] or self.points[-1][2] != vals[6]:
                self.points.append((vals[4], vals[5], vals[6]))

            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            self.drawXYPoints(imgXY)
            self.drawXZPoints(imgXZ)
            cv2.imshow("imgXY",imgXY)
            cv2.imshow("imgXZ",imgXZ)
            cv2.waitKey(1)

        # Destroying all the windows
        cv2.destroyAllWindows()

        # Stop recording
        rec.stop()
        me.streamoff()


    def runDroneWebcam(self):
        """
        IS ALWAYS isWebcam=True, DRONE FLY, VIDEO RECORDED FROM DRONE AND FROM WEBCAM,
        DETECTION TRAJECTORY FROM WEBCAM
        """

        self.isWebcam = True

        me = tello.Tello()
        me.connect()
        print(me.get_battery())

        # Path for save things
        idx = self.setLastIdx()
        path = f"{self.VIDEO_DIR_PATH}\\{idx}"

        fullControll = fullControllModule.FullControll()

        # Reset values
        fullControll.autoSet(path, isWebcam=self.isWebcam, resize=False, showPlot=False)

        # Get the stream image
        me.streamon() # to get the stream image
        time.sleep(3)
       
        # Start rec video
        rec = recVid.recordVideo(me, f"{path}_droneCamera")
        rec.run()

        # Takeoff
        me.takeoff()
        time.sleep(3) 

        # Fly up a bit
        me.send_rc_control(0, 0, 20, 0)

        # Get data from hand
        resTraj = fullControll.run(me)

        # Export original data as CSV
        pd.DataFrame(np.array(resTraj)).to_csv(f"{path}_original.csv", index=False, header=None)

        # Get resolution
        height, width = fullControll.getResolution()

        velocities = self.normalizeData(resTraj, height, width, path=path, log=False)

        while self.flag:

            # Control with detected trajectory
            vals = self.getKeyboardInput2(velocities)

            imgXY = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
            imgXZ = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
            
            if self.points[-1][0] != vals[4] or self.points[-1][1] != vals[5] or self.points[-1][2] != vals[6]:
                self.points.append((vals[4], vals[5], vals[6]))

            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            self.drawXYPoints(imgXY)
            self.drawXZPoints(imgXZ)
            cv2.imshow("imgXY",imgXY)
            cv2.imshow("imgXZ",imgXZ)
            cv2.waitKey(1)

        # Save the real trajectory
        cv2.imwrite(f"{path}_realtrajXY.png", imgXY)
        cv2.imwrite(f"{path}_realtrajXZ.png", imgXZ)

        # Destroying all the windows
        cv2.destroyAllWindows()

        # Stop recording
        rec.stop()
        me.streamoff()

        # Close kp
        fullControll.closekp()


    def test(self):
        """
        THIS IS TEST, IS ALWAYS isWebcam=True, DRONE DON'T FLY, VIDEO RECORDED FROME DRONE AND FROM WEBCAM
        DETECTION TRAJECTORY FROM WEBCAM
        """
        self.isWebcam = True

        me = tello.Tello()
        me.connect()
        print(me.get_battery())

        # Path for save things
        idx = self.setLastIdx()
        path = f"{self.VIDEO_DIR_PATH}\\{idx}"

        fullControll = fullControllModule.FullControll()

        # Reset values
        fullControll.autoSet(path, isWebcam=self.isWebcam, resize=False, showPlot=False)

        # Get the stream image
        print("Get the stream image...")
        me.streamon()
        time.sleep(3)
        
        # Start rec video
        rec = recVid.recordVideo(me, f"{path}_droneCamera")
        rec.run()

        print("Takeoff...")
        time.sleep(3)

        # Fly up a bit
        print("Fly up a bit...")
   
        # Get data from hand
        resTraj = fullControll.run(me)

        # Export original data as CSV
        pd.DataFrame(np.array(resTraj)).to_csv(f"{path}_original.csv", index=False, header=None)

        # Get resolution
        height, width = fullControll.getResolution()

        velocities = self.normalizeData(resTraj, height, width, path=path, log=False)

        while self.flag:

            # Control with detected trajectory
            vals = self.getKeyboardInput2(velocities)

            imgXY = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
            imgXZ = np.zeros((1000,1000,3), dtype=np.uint8) # 0 - 256
            
            if self.points[-1][0] != vals[4] or self.points[-1][1] != vals[5] or self.points[-1][2] != vals[6]:
                self.points.append((vals[4], vals[5], vals[6]))

            me.send_rc_control(vals[0], vals[1], vals[2], vals[3])
            self.drawXYPoints(imgXY)
            self.drawXZPoints(imgXZ)
            cv2.imshow("imgXY",imgXY)
            cv2.imshow("imgXZ",imgXZ)
            cv2.waitKey(1)

        # Save the real trajectory
        cv2.imwrite(f"{path}_realtrajXY.png", imgXY)
        cv2.imwrite(f"{path}_realtrajXZ.png", imgXZ)

        # Destroying all the windows
        cv2.destroyAllWindows()

        # Stop recording
        rec.stop()
        me.streamoff()

        # Close kp
        fullControll.closekp()


def main():

    # isWebcam=True IF YOU WANT TO USE WEBCAM
    kc = keyboardControl()

    # IS ALWAYS isWebcam=False, DRONE FLY, VIDEO RECORDED FROM DRONE,
    # DETECTION TRAJECTORY FROM DRONE
    #kc.runJustDrone()

    # IS ALWAYS isWebcam=True, DRONE FLY, VIDEO RECORDED FROM DRONE AND FROM WEBCAM,
    # DETECTION TRAJECTORY FROM WEBCAM
    #kc.runDroneWebcam()

    # THIS IS TEST, IS ALWAYS isWebcam=True, DRONE DON'T FLY, VIDEO RECORDED FROME DRONE AND FROM WEBCAM
    # DETECTION TRAJECTORY FROM WEBCAM
    kc.runJustDrone()


if __name__ == "__main__":
    
    main()