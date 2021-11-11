import time
import cv2
from numba import jit
import numpy as np
import pdb

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class dynamic3dDrawTrajectory():

    def __init__(self):
        plt.ion()
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')

    def run(self, xdata, ydata, zdata):
        #print(xdata, ydata, zdata)

        '''
        # if we want don't take care about z-depth
        ydata = []
        for x in range(len(xdata)):
            ydata.append(0)
        '''

        self.ax.plot3D(xdata, ydata, zdata, 'gray')
        plt.draw()
        plt.pause(0.002)
        self.ax.cla()
           
        #print(xdata, ydata, zdata)



def main():
    '''
    https://stackoverflow.com/questions/5179589/continuous-3d-plotting-i-e-figure-update-using-python-matplotlib
    https://stackoverflow.com/questions/10944621/dynamically-updating-plot-in-matplotlib
    '''

    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xdata = []
    ydata = []
    zdata = []
    for x in np.arange(0,20,0.5):
        xdata.append(x)
        ydata.append(0)
        zdata.append(np.exp(-x**2)+10*np.exp(-(x-7)**2))
        ax.plot3D(xdata, ydata, zdata, 'gray')
        plt.draw()
        plt.pause(0.5)
        ax.cla()

if __name__ == "__main__":
    main()