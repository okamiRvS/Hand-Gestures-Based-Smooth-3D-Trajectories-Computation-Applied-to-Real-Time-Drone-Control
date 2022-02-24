import pandas as pd
from matplotlib import pyplot as plt
 
TARGET = ['backward', 'detect', 'down', 'forward', 'land', 'left', 'ok', 'right', 'stop', 'up']
#val = [0.93, 0.93, 0.92, 0.98, 0.95, 0.94, 1, 0.85, 0.86, 0.98] # feat sel
#val = [0.86, 0.89, 0.83, 0.98, 0.86, 0.94, 1, 0.89, 0.89, 0.94] # pca
val = [0.97, 0.92, 0.96, 1, 0.90, 0.98, 1, 0.93, 1, 0.94] # allfeat
  
# Figure Size
fig, ax = plt.subplots(figsize =(10, 6))
 
# Horizontal Bar Plot
ax.barh(TARGET, val, color="maroon")

plt.xlabel('F1-Score')
plt.ylabel('Class')

for bar, alpha in zip(ax.containers[0], val):
    bar.set_alpha(alpha)
 
# Remove axes splines
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
 
# Remove x, y Ticks
ax.xaxis.set_ticks_position('none')
ax.yaxis.set_ticks_position('none')
 
# Add padding between axes and labels
ax.xaxis.set_tick_params(pad = 5)
ax.yaxis.set_tick_params(pad = 10)
 
# Add x, y gridlines
ax.grid(b = True, color ='grey',
        linestyle ='-.', linewidth = 0.5,
        alpha = 0.2)
 
# Show top values
ax.invert_yaxis()
 
# Add annotation to bars
for i in ax.patches:
    plt.text(i.get_width()+0.02, i.get_y()+0.5,
             str(round((i.get_width()), 2)),
             fontsize = 10, fontweight ='bold',
             color ='grey')

# Show Plot
plt.show()