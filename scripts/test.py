from djitellopy import tello
from time import time
import cv2
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import pdb

# Define some points:
theta = np.linspace(-3, 2, 40)
points = np.vstack( (np.vstack( (np.cos(theta), np.sin(theta)) ), np.tanh(theta)) ).T

# add some noise:
points = points + 0.05*np.random.randn(*points.shape)

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Build a list of the spline function, one for each dimension:
splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

# Computed the spline for the asked distances:
alpha = np.linspace(0, 1, 75)
#points_fitted = np.vstack( (splines[0](alpha), splines[1](alpha), splines[2](alpha)) ).T

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')


# vec=(1,0,0) in homogeneous coordinates
vec = np.array([1,0,0,1])
theta = 90 * np.pi / 180
Matrix3dRotationX = np.array([[1, 0, 0, 0], [0, np.cos(theta), np.sin(theta), 0], [0, -np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])
Matrix3dRotationY = np.array([[np.cos(theta), 0, -np.sin(theta), 0], [0, 1, 0, 0], [np.sin(theta), 0, np.cos(theta), 0], [0, 0, 0, 1]])
Matrix3dRotationZ = np.array([[np.cos(theta), -np.sin(theta), 0, 0], [np.sin(theta), np.cos(theta), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

tmp = (Matrix3dRotationX @ vec.T).T
tmp2 = (Matrix3dRotationY @ tmp.T).T
pdb.set_trace()
tmp3 = (Matrix3dRotationZ @ tmp2.T).T

ax.quiver(splines[0](alpha),
        splines[1](alpha),
        splines[2](alpha),
        splines[0](alpha)+1000,
        splines[1](alpha)+1,
        splines[2](alpha)+1000,
        length=0.1,
        normalize=True)
plt.draw()
plt.show()