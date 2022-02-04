from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pdb


def Ridge2D():

    #Define input array with angles from 60deg to 300deg converted to radians
    x = np.array([i*np.pi/180 for i in range(60,300,4)])
    np.random.seed(10)  #Setting seed for reproducibility
    y = np.sin(x) + np.random.normal(0,0.15,len(x))
    data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
    plt.plot(data['x'], data['y'], '.')
    plt.show()

    # Let’s try to estimate the sine function using polynomial regression
    # with powers of x from 1 to 15. Let’s add a column for each power
    # upto 15 in our dataframe.
    for i in range(2,16):  #power of 1 is already there
        colname = 'x_%d'%i      #new var will be x_power
        data[colname] = data['x']**i
    print(data.head())

    # Define the predictors
    predictors=["x"]
    predictors.extend([f"x_{i}" for i in range(2,16)])

    # Fit!
    clf = Ridge(alpha=1.0)
    clf.fit(data[predictors], data.y)

    # Predict new data...
    y_pred = clf.predict(data[predictors])
    plt.plot(data['x'], data['y'], '.')
    plt.plot(x, y_pred)
    plt.show()


def Ridge3DEasy():

    #Setting seed for reproducibility
    np.random.seed(10)  

    # Data for three-dimensional scattered points
    xdata = np.linspace(0, 100, 100, dtype=np.int)
    ydata = xdata + 0.9 * np.random.normal(0, 10, len(xdata))
    zdata = np.zeros_like(xdata)
    data = pd.DataFrame( np.column_stack( [xdata, ydata, zdata] ),columns=['x', 'y', 'z'])

    # Let’s try to estimate the function using polynomial regression
    # with powers of x and y from 1 to 15. Let’s add a column for each power
    # upto 15 in our dataframe.
    n_features = 6
    for i in range(2,n_features):  #power of 1 is already there

        # for x
        colname = f"x_{i}"
        data[colname] = data["x"]**i

    print(data.head())

    # Define the predictors
    predictors=["x"]
    for i in range(2,n_features):
        predictors.extend([f"x_{i}"])

    # Fit!
    clf = Ridge(alpha=1e-10)
    clf.fit(data[predictors], data[["y", "z"]])

    # Predict new data...
    pred = clf.predict(data[predictors])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter3D(data.x, data.y, data.z, cmap='Greens')
    ax.plot3D(data.x, pred[:,0], pred[:,1], 'orange')
    plt.show()


def Ridge3D():

    #Setting seed for reproducibility
    np.random.seed(10)  

    # Data for three-dimensional scattered points
    zdata = np.linspace(0, 15, 100)
    xdata = np.sin(zdata) + np.random.normal(0, 0.1, len(zdata))
    ydata = np.cos(zdata) + np.random.normal(0, 0.1, len(zdata))
    data = pd.DataFrame( np.column_stack( [xdata, ydata, zdata] ),columns=['x', 'y', 'z'])

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.plot3D(xdata, ydata, zdata, 'black')    

    # Fit!
    distance = np.cumsum( np.sqrt(np.sum( np.diff(data, axis=0)**2, axis=1 )) )
    distance = np.insert(distance, 0, 0)/distance[-1]
    distance = pd.DataFrame( np.column_stack( [distance, xdata, ydata, zdata] ),columns=["dist", "x", "y", "z"])

    n_features = 10
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
    splines = []
    clf = Ridge(alpha=1e-10)
    splines.append(clf.fit(distance[predictors], distance[ ["x"] ]))
    clf = Ridge(alpha=1e-10)
    splines.append(clf.fit(distance[predictors], distance[ ["y"] ]))
    clf = Ridge(alpha=1e-10)
    splines.append(clf.fit(distance[predictors], distance[ ["z"] ]))

    alpha = np.linspace(0, 1, 100).reshape(-1,1)

    out = []
    for spline in splines:
        out.append(spline.predict(distance[predictors]))

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter3D(data.x, data.y, data.z, cmap='black')
    ax.plot3D(out[0].T[0], out[1].T[0], out[2].T[0], 'orange')
    plt.show()


def test():
    n_samples, n_features = 100, 1

    rng = np.random.RandomState(0)

    y = rng.randn(n_samples)
    X = rng.randn(n_samples, n_features)
    clf = Ridge(alpha=1.0)
    clf.fit(X, y)

    plt.plot(X, y)
    plt.show()

def main():

    #Ridge2D()
    #Ridge3DEasy()
    Ridge3D()


if __name__ == "__main__":
    
    main()