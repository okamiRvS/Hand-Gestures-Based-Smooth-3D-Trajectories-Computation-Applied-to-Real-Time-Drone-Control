from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
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
    #https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
    #https://www.statology.org/adjusted-r-squared-in-python/
    #https://towardsdatascience.com/what-are-the-best-metrics-to-evaluate-your-regression-model-418ca481755b
    #https://towardsdatascience.com/20-popular-machine-learning-metrics-part-1-classification-regression-evaluation-metrics-1ca3e282a2ce
    #https://www.kaggle.com/residentmario/model-fit-metrics/notebook
    #https://scikit-learn.org/stable/modules/model_evaluation.html#r2-score
    # for 1 gesto con k=10 gradi diversi e salvare r^2, MSE e RMSE

    #Setting seed for reproducibility
    np.random.seed(10)  

    # Data for three-dimensional scattered points
    zdata = np.linspace(0, 15, 100)
    xdata = np.sin(zdata) + np.random.normal(0, .3, len(zdata))
    ydata = np.cos(zdata) + np.random.normal(0, .3, len(zdata))
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
    data["dist"] = distance

    n_features = 10
    for i in range(2,n_features):  #power of 1 is already there
        colname = f"dist_{i}"
        data[colname] = data["dist"]**i

    print(data.head())
    plt.show()

    # Define the predictors
    predictors=["dist"]
    for i in range(2,n_features):
        predictors.extend([f"dist_{i}"])

    # Build a list of the spline function, one for each dimension:
    splines = []
    clf = Ridge(alpha=1e-10)
    splines.append(clf.fit(data[predictors], data[ ["x"] ]))
    print(f'adj_r2_score x: { 1 - (1-clf.score(data[predictors], data[ ["x"] ]))*(len(data[ ["x"] ])-1)/(len(data[ ["x"] ])-data[predictors].shape[1]-1) }')
    clf = Ridge(alpha=1e-10)
    splines.append(clf.fit(data[predictors], data[ ["y"] ]))
    print(f'adj_r2_score y: { 1 - (1-clf.score(data[predictors], data[ ["y"] ]))*(len(data[ ["y"] ])-1)/(len(data[ ["y"] ])-data[predictors].shape[1]-1) }')
    clf = Ridge(alpha=1e-10)
    splines.append(clf.fit(data[predictors], data[ ["z"] ]))
    print(f'adj_r2_score z: { 1 - (1-clf.score(data[predictors], data[ ["z"] ]))*(len(data[ ["z"] ])-1)/(len(data[ ["z"] ])-data[predictors].shape[1]-1) }')
    
    out = []
    for spline in splines:
        out.append(spline.predict(data[predictors]))

    print(f'r2_score x: {r2_score(data[ ["x"] ], out[0])}')
    print(f'r2_score y: {r2_score(data[ ["y"] ], out[1])}')
    print(f'r2_score z: {r2_score(data[ ["z"] ], out[2])}')
    print(f'RMSE x: {np.sqrt(mean_squared_error(data[ ["x"] ], out[0]))}')
    print(f'RMSE y: {np.sqrt(mean_squared_error(data[ ["y"] ], out[1]))}')
    print(f'RMSE z: {np.sqrt(mean_squared_error(data[ ["z"] ], out[2]))}')

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    ax.scatter3D(data.x, data.y, data.z, cmap='black')
    ax.plot3D(out[0].T[0], out[1].T[0], out[2].T[0], 'orange')
    plt.show()

def main():

    #Ridge2D()
    #Ridge3DEasy()
    Ridge3D()
    

if __name__ == "__main__":
    
    main()