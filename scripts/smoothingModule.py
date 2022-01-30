import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd
import copy
import pdb


class smoothing():

    def __init__(self, skipEveryNpoints):

        self.skipEveryNpoints = skipEveryNpoints

    def setPoints(self, xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed):

        # Computed the spline for the asked distances:
        subdivision = len(xdata) * 3
        self.alpha = np.linspace(0, 1, subdivision)

        coord = np.array([xdata, ydata, zdata]).T
        
        orientation = np.array([rolldata, yawdata, pitchdata]).T
        dtime = np.array([dtime], dtype=np.float64)
        speed = np.array([speed])

        tmpTime = self.smoothData(dtime)[0]
        
        self.data = {
            "position": self.Ridge3D(coord), #self.smoothData(coord), 
            "orientation": self.Ridge3D(orientation), #self.smoothData(orientation),
            "time": [np.where(tmpTime < 0, 0, tmpTime)][0], # to have only positive values
            "speed": self.smoothData(speed)[0]
        }

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')

        # ax.scatter(self.coordData[0], self.coordData[1], self.coordData[2], s=self.speedData[0],c=self.speedData[0])
        # #ax.scatter(points[:,0], points[:,1], points[:,2], c='grey')
        # plt.draw()
        # plt.show()
        # pdb.set_trace()


    def Ridge3D(self, po):

        # Data for three-dimensional scattered points
        data = pd.DataFrame( np.column_stack( [po[:,0], po[:,1], po[:,2]] ),columns=['x', 'y', 'z'])
        test = pd.DataFrame( np.column_stack( [self.alpha] ),columns=['t'])

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # ax.plot3D(po[:,0], po[:,1], po[:,2], 'black') 
        # plt.show()   

        # Fit!
        distance = np.cumsum( np.sqrt(np.sum( np.diff(data, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        distance = pd.DataFrame( np.column_stack( [distance, po[:,0], po[:,1], po[:,2]] ),columns=["dist", "x", "y", "z"])

        n_features = 7
        for i in range(2,n_features):  #power of 1 is already there
            colname = f"dist_{i}"
            distance[colname] = distance["dist"]**i

        print(distance.head())
        plt.show()

        # Define the predictors
        predictors=["dist"]
        for i in range(2,n_features):
            predictors.extend([f"dist_{i}"])

        # Build a list of the spline function, one for each dimension:
        rid = []
        clf = Ridge(alpha=1e-10)
        rid.append(clf.fit(distance[predictors], distance[ ["x"] ]))
        clf = Ridge(alpha=1e-10)
        rid.append(clf.fit(distance[predictors], distance[ ["y"] ]))
        clf = Ridge(alpha=1e-10)
        rid.append(clf.fit(distance[predictors], distance[ ["z"] ]))

        alpha = copy.deepcopy(self.alpha.reshape(-1,1))
        alpha = pd.DataFrame( np.column_stack( [alpha] ),columns=["alpha"])

        for i in range(2,n_features):  #power of 1 is already there
            colname = f"alpha_{i}"
            alpha[colname] = alpha["alpha"]**i

        predictors=["alpha"]
        for i in range(2,n_features):
            predictors.extend([f"alpha_{i}"])

        out = []
        for ri in rid:
            out.append(ri.predict(alpha[predictors]))

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(data.x, data.y, data.z, cmap='black')
        # ax.plot3D(out[0].T[0], out[1].T[0], out[2].T[0], 'orange')
        # plt.show()
        # pdb.set_trace()

        return out


    def smoothCalculation(self):

        return self.data["position"][0], self.data["position"][1], self.data["position"][2], self.data["orientation"][0], self.data["orientation"][1], self.data["orientation"][2], self.data["time"], self.data["speed"]


    def skipEveryNpointsFunc(self):

        # skip each n points, to have a better view of data
        self.data["position"][0] = self.data["position"][0][::self.skipEveryNpoints]
        self.data["position"][1] = self.data["position"][1][::self.skipEveryNpoints]
        self.data["position"][2] = self.data["position"][2][::self.skipEveryNpoints]

        self.data["orientation"][0] = self.data["orientation"][0][::self.skipEveryNpoints]
        self.data["orientation"][1] = self.data["orientation"][1][::self.skipEveryNpoints]
        self.data["orientation"][2] = self.data["orientation"][2][::self.skipEveryNpoints]

        self.data["time"] = self.data["time"][::self.skipEveryNpoints]

        self.data["speed"] = self.data["speed"][::self.skipEveryNpoints]


    def smoothData(self, data):

        # Linear length along the line:
        if data.shape[0] == 1:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(data, axis=1)**2, axis=0 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]

            spline = UnivariateSpline(distance, data, k=3)

            return [spline(self.alpha)]
        else:
            distance = np.cumsum( np.sqrt(np.sum( np.diff(data, axis=0)**2, axis=1 )) )
            distance = np.insert(distance, 0, 0)/distance[-1]
            
            # Build a list of the spline function, one for each dimension:
            splines = [UnivariateSpline(distance, coords, k=3, s=.2) for coords in data.T]

        out = []
        for spline in splines:
            out.append(spline(self.alpha))

        return out


def main():

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


if __name__ == "__main__":

    main()