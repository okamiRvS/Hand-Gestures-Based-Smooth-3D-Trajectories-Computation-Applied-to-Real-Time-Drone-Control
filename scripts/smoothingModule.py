import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
import numpy as np
import pdb

'''
rng = np.random.default_rng()
x = np.linspace(-3, 3, 50)
y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
plt.plot(x, y, 'ro', ms=5)

spl = UnivariateSpline(x, y)
xs = np.linspace(-3, 3, 1000)
plt.plot(xs, spl(xs), 'g', lw=3)

spl.set_smoothing_factor(0.5)
plt.plot(xs, spl(xs), 'b', lw=3)
plt.show()
'''

'''
# Define some points:
theta = np.linspace(-3, 2, 40)
points = np.vstack( (np.cos(theta), np.sin(theta)) ).T
#points = np.vstack( (np.vstack( (np.cos(theta), np.sin(theta)) ), np.tanh(theta)) ).T

# add some noise:
points = points + 0.05*np.random.randn(*points.shape)

# Linear length along the line:
distance = np.cumsum( np.sqrt(np.sum( np.diff(points, axis=0)**2, axis=1 )) )
distance = np.insert(distance, 0, 0)/distance[-1]

# Build a list of the spline function, one for each dimension:
splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in points.T]

# Computed the spline for the asked distances:
alpha = np.linspace(0, 1, 75)
points_fitted = np.vstack( (splines[0](alpha), splines[1](alpha)) ).T

# Graph:
plt.plot(*points.T, 'ok', label='original points')
plt.plot(*points_fitted.T, '-r', label='fitted spline k=3, s=.2')
plt.axis('equal'); plt.legend(); plt.xlabel('x'); plt.ylabel('y')
plt.show()
'''

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

ax.scatter(splines[0](alpha), splines[1](alpha), splines[2](alpha), c=splines[2](alpha))
ax.scatter(points[:,0], points[:,1], points[:,2], c='grey')
plt.draw()
plt.show()