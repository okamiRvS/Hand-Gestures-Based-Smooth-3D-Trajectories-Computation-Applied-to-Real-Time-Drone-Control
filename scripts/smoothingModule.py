import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.interpolate import UnivariateSpline
import numpy as np
import pandas as pd
import copy
import pdb


class smoothing():

    def __init__(self, skipEveryNpoints, path=None):

        self.skipEveryNpoints = skipEveryNpoints
        self.path = path

    def setPoints(self, xdata, ydata, zdata, rolldata, yawdata, pitchdata, dtime, speed):

        # Computed the spline for the asked distances:
        subdivision = len(xdata) * 5
        self.alpha = np.linspace(0, 1, subdivision)

        coord = np.array([xdata, ydata, zdata]).T
        
        orientation = np.array([rolldata, yawdata, pitchdata]).T
        dtime = np.array([dtime], dtype=np.float64)
        speed = np.array([speed])

        tmpTime = self.smoothData(dtime)[0]
        
        self.Ridge3DForMetrics(coord) ########## JUST FOR TEST

        self.data = {
            "position": self.Ridge3D(coord), #self.smoothData(coord)
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
        data["dist"] = distance

        n_features = 7
        for i in range(2,n_features):  #power of 1 is already there
            colname = f"dist_{i}"
            data[colname] = data["dist"]**i

        #print(data.head())

        # Define the predictors
        predictors=["dist"]
        for i in range(2,n_features):
            predictors.extend([f"dist_{i}"])

        # Build a list of the spline function, one for each dimension:
        rid = []
        clf = Ridge(alpha=1e-10)
        rid.append(clf.fit(data[predictors], data[ ["x"] ]))
        clf = Ridge(alpha=1e-10)
        rid.append(clf.fit(data[predictors], data[ ["y"] ]))
        clf = Ridge(alpha=1e-10)
        rid.append(clf.fit(data[predictors], data[ ["z"] ]))

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
            out.append(ri.predict(alpha[predictors]).T[0])

        # fig = plt.figure()
        # ax = plt.axes(projection='3d')
        # ax.scatter3D(data.x, data.y, data.z, cmap='black')
        # ax.plot3D(out[0], out[1], out[2], 'orange')
        # plt.show()
        # pdb.set_trace()

        return out


    def Ridge3DForMetrics(self, po):

        # This function it is useful just to get information about goodness of fit

        # Data for three-dimensional scattered points
        data = pd.DataFrame( np.column_stack( [po[:,0], po[:,1], po[:,2]] ),columns=['x', 'y', 'z'])
        test = pd.DataFrame( np.column_stack( [self.alpha] ),columns=['t'])

        alpha = copy.deepcopy(self.alpha.reshape(-1,1))
        alpha = pd.DataFrame( np.column_stack( [alpha] ),columns=["alpha"])

        # Fit!
        distance = np.cumsum( np.sqrt(np.sum( np.diff(data, axis=0)**2, axis=1 )) )
        distance = np.insert(distance, 0, 0)/distance[-1]
        data["dist"] = distance

        n_features = 10
        predictors_train=["dist"]
        col = ['degree',
                'adj_r2_score_x',
                'adj_r2_score_y',
                'adj_r2_score_z', 
                'r2_score_x', 
                'r2_score_y', 
                'r2_score_z',
                'MSE_x',
                'MSE_y', 
                'MSE_z', 
                'RMSE_x', 
                'RMSE_y', 
                'RMSE_z']
        df = pd.DataFrame(columns=col)
        
        fig = plt.figure(1)
        ax = fig.add_subplot(111, projection='3d')      
        for i in range(2,n_features):

            colname = f"dist_{i}"
            data[colname] = data["dist"]**i
            predictors_train.extend([f"dist_{i}"])

            # Build a list of the ridge function, one for each dimension (x,y,z):
            rid = []
            clf = Ridge(alpha=1e-10)
            rid.append(clf.fit(data[predictors_train], data[ ["x"] ]))
            adjr2x = 1 - (1-clf.score(data[predictors_train], data[ ["x"] ]))*(len(data[ ["x"] ])-1)/(len(data[ ["x"] ])-data[predictors_train].shape[1]-1)
            
            clf = Ridge(alpha=1e-10)
            rid.append(clf.fit(data[predictors_train], data[ ["y"] ]))
            adjr2y = 1 - (1-clf.score(data[predictors_train], data[ ["y"] ]))*(len(data[ ["y"] ])-1)/(len(data[ ["y"] ])-data[predictors_train].shape[1]-1)
            
            clf = Ridge(alpha=1e-10)
            rid.append(clf.fit(data[predictors_train], data[ ["z"] ]))
            adjr2z = 1 - (1-clf.score(data[predictors_train], data[ ["z"] ]))*(len(data[ ["z"] ])-1)/(len(data[ ["z"] ])-data[predictors_train].shape[1]-1)

            # Metric on train
            out = []
            for ri in rid:
                out.append(ri.predict(data[predictors_train]))

            r2_x = r2_score(data[ ["x"] ], out[0])
            r2_y = r2_score(data[ ["y"] ], out[1])
            r2_z = r2_score(data[ ["z"] ], out[2])
            mse_x = mean_squared_error(data[ ["x"] ], out[0])
            mse_y = mean_squared_error(data[ ["y"] ], out[1])
            mse_z = mean_squared_error(data[ ["z"] ], out[2])
            rmse_x = np.sqrt(mean_squared_error(data[ ["x"] ], out[0]))
            rmse_y = np.sqrt(mean_squared_error(data[ ["y"] ], out[1]))
            rmse_z = np.sqrt(mean_squared_error(data[ ["z"] ], out[2]))

            df.loc[i] = [i, r2_x, r2_y, r2_z, adjr2x, adjr2y, adjr2z, mse_x, mse_y, mse_z, rmse_x, rmse_y, rmse_z]
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            ax.set_xlim3d([0, 1])
            ax.set_ylim3d([1, -1])
            ax.set_zlim3d([0, 1])
            #ax.set_title(f"Metric degree: {i}")
            ax.scatter3D(data.x, data.y, data.z, cmap='black')
            ax.plot3D(out[0].T[0], out[1].T[0], out[2].T[0], 'orange')
            fig.savefig(f"{self.path}_deg_{i}")
            ax.clear()
  
        # Export normalized data as CSV
        pd.DataFrame(df).to_csv(f"{self.path}_ridgeMetric.csv", index=False, header=col)


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
            splines = [UnivariateSpline(distance, coords, k=5, s=.2) for coords in data.T]

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