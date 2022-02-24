TARGET = ['backward', 'detect', 'down', 'forward', 'land', 'left', 'ok', 'right', 'stop', 'up']
precisionAllFeatures = [0.96,0.86,0.92,0.97,1,1,1,0.83,0.84,1]

# libraries
import numpy as np
import matplotlib.pyplot as plt
import pdb
 
# width of the bars
barWidth = 0.3

# ############################################# FEATURE SELECTED
# # Choose the height of the blue bars
# bars1 = [0.96 ,0.86, 0.92, 0.97, 1, 1, 1, 0.83, 0.84, 1] # precision
 
# # Choose the height of the cyan bars
# bars2 = [0.90, 1, 0.92, 1, 0.9, 0.88, 1, 0.87, 0.89, 0.96] # recall
################################################################

# ############################################# PCA
# Choose the height of the blue bars
# bars1 = [1, 0.88, 0.75, 0.97, 0.83, 1, 1, 0.91, 0.85, 0.96] # precision
 
# # Choose the height of the cyan bars
# bars2 = [0.76, 0.9, 0.92, 1, 0.9, 0.88, 1, 0.87, 0.94, 0.82] # recall
################################################################
 
# ############################################# All Feat
# Choose the height of the blue bars
bars1 = [0.94, 0.91, 0.96, 1, 0.9, 0.96, 1, 0.95, 1, 1] # precision
 
# Choose the height of the cyan bars
bars2 = [1, 0.94, 0.96, 1, 0.9, 1, 1, 0.91, 1, 0.88] # recall
################################################################

# The x position of bars
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]

# plotting    
f, ax = plt.subplots(figsize=(8, 8))
 
# Create blue bars
plt.bar(r1, bars1, width = barWidth, color = 'blue', edgecolor = 'black', capsize=7, label='Precision')
 
# Create cyan bars
plt.bar(r2, bars2, width = barWidth, color = 'cyan', edgecolor = 'black', capsize=7, label='Recall')
 
# general layout
plt.xticks([r + barWidth for r in range(len(bars1))], TARGET, rotation=70)
plt.ylabel('height')
plt.legend(loc='lower right')
 
# Show graphic
plt.show()