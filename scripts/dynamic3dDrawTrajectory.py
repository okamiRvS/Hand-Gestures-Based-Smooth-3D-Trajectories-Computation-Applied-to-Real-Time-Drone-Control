import time
import cv2
from numba import jit
import numpy as np
import pdb

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dynamic3dDrawTrajectory():

    def __init__(self, skipEveryNpoints=2):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
    
        self.clean()

        self.skipEveryNpoints = skipEveryNpoints
        self.nextTime = time.time()

        # initializaztion of cbar
        self.cbar = self.fig.colorbar(
            self.ax.scatter([], [], [], c=[]),
            shrink=0.5, aspect=5
        )

        self.start = self.ax.text3D(0, 0, 0, "", zdir='x', size=10, zorder=1, color='black') 
        self.end = self.ax.text3D(0, 0, 0, "", zdir='x', size=10, zorder=1, color='black') 

    def run(self, xdata, ydata, zdata, speed):

        currentTime = time.time()
        
        if currentTime > self.nextTime:
            self.nextTime = currentTime + 2

            # skip each n points, to have a better view of data
            xdata = xdata[::self.skipEveryNpoints]
            ydata = ydata[::self.skipEveryNpoints]
            zdata = zdata[::self.skipEveryNpoints]
            speed = speed[::self.skipEveryNpoints]

            #print(xdata, ydata, zdata)
            #print(speed)
            
            #self.ax.plot3D(xdata, ydata, zdata, c='gray')
            surf = self.ax.scatter(xdata, ydata, zdata, s=speed,c=speed, vmin = 2, vmax = 10)

            lenList = len(xdata)
            if lenList > 1:
                # speed[0] is zero so the point has size zero, so its't not visibile in the graph

                # https://stackoverflow.com/questions/63546097/3d-curved-arrow-in-python
                # https://gist.github.com/WetHat/1d6cd0f7309535311a539b42cccca89c
                self.start.set_position_3d( (xdata[1], ydata[1], zdata[1]) )
                self.start.set_text("START")

                self.end.set_position_3d( (xdata[lenList - 1], ydata[lenList - 1], zdata[lenList - 1]) )
                self.end.set_text("END")

            # Remove color bar which maps values to colors.
            self.cbar.remove()

            # Add a color bar which maps values to colors.
            self.cbar = self.fig.colorbar(surf, shrink=0.5, aspect=5)

            plt.draw()
            #plt.pause(0.5)
  
            
            #print(xdata, ydata, zdata)

    def clean(self):
        # delete everything
        self.ax.clear()

        # Label each axis
        self.ax.set_xlabel('X Label')
        self.ax.set_ylabel('Y Label')
        self.ax.set_zlabel('Z Label')

        # Set each axis limits
        self.ax.set_xlim3d([0, 1])
        self.ax.set_ylim3d([-1, 1])
        self.ax.set_zlim3d([0, 1])

def main():
    '''
    https://matplotlib.org/2.0.2/mpl_toolkits/mplot3d/tutorial.html
    https://stackoverflow.com/questions/5179589/continuous-3d-plotting-i-e-figure-update-using-python-matplotlib
    https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib
    '''

    #plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xdata = []
    ydata = []
    zdata = []
    for x in np.arange(0,20,0.5):
        # Label each axis
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        # Set each axis limits
        #ax.set_xlim3d([0, 1])
        #ax.set_ylim3d([0, 1])
        #ax.set_zlim3d([0, 1])

        xdata.append(x)
        ydata.append(0)
        zdata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
        #ax.plot3D(xdata, ydata, zdata, 'gray')
        ax.scatter(xdata, ydata, zdata, c=zdata)
        plt.draw()
        plt.pause(0.5)
        ax.clear()

if __name__ == "__main__":
    main()