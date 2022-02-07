from matplotlib import markers
import pandas as pd
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pdb

df = pd.read_csv('scripts/test_scripts/10_ridgeMetric.csv')

print(df.head()) 

#########
# Adjusted R-Squared
thisLegend = []

plt.ylabel("Adjusted R-Squared")
plt.xlabel("Degree")

plt.plot(df["degree"], df["adj_r2_score_x"], 'r-', marker='<')
thisLegend.append(f"Adjusted R-Squared x")

plt.plot(df["degree"], df["adj_r2_score_y"], 'g-', marker='P')
thisLegend.append(f"Adjusted R-Squared y")

plt.plot(df["degree"], df["adj_r2_score_z"], 'b-', marker='8')
thisLegend.append(f"Adjusted R-Squared z")

plt.legend(thisLegend)
plt.show()


#########
#R-Squared
thisLegend = []
plt.ylabel("R-Squared")
plt.xlabel("Degree")

plt.plot(df["degree"], df["r2_score_x"], 'r-.', marker='v')
thisLegend.append(f"R-Squared x")

plt.plot(df["degree"], df["r2_score_y"], 'g-.', marker='*')
thisLegend.append(f"R-Squared y")

plt.plot(df["degree"], df["r2_score_z"], 'b-.', marker='<')
thisLegend.append(f"R-Squared z")

plt.legend(thisLegend)
plt.show()

#########
#MSE
thisLegend = []
plt.ylabel("RMSE")
plt.xlabel("Degree")

plt.plot(df["degree"], df["RMSE_x"], 'r--', marker='^')
thisLegend.append(f"RMSE x")

plt.plot(df["degree"], df["RMSE_y"], 'g--', marker='x')
thisLegend.append(f"RMSE y")

plt.plot(df["degree"], df["RMSE_z"], 'b--', marker='s')
thisLegend.append(f"RMSE z")

plt.legend(thisLegend)
plt.show()