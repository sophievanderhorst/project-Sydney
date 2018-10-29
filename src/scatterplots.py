#!/usr/bin/env python

"""
I use this script to make scatterplots that show measurement ratios of 
low temperature vs high temperatures. 
"""

__author__ = "Sophie van der Horst"
__version__ = "1.0 (25.10.2018)"
__email__ = "sophie.vanderhorst@wur.nl"


import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.patches as mpatches
import os

with open("../data/processed/results_LH.csv", newline='') as myFile:  
    df_results_LH = pd.read_csv(myFile)
    for row in df_results_LH:
        print(row) 
        
with open("../data/processed/results_LH.csv", newline='') as myFile:  
    df_results_SH = pd.read_csv(myFile)
    for row in df_results_SH:
        print(row) 

with open("../data/processed/results_LH.csv", newline='') as myFile:  
    df_results_NEE = pd.read_csv(myFile)
    for row in df_results_NEE:
        print(row) 

    # creating scatterplots for temperatures below mean vs temperatures above mean. For LH, SH and NEE.
fig, ax = plt.subplots(ncols = 2, nrows = 3, figsize = (10,15))
ax[0,0].scatter(df_results_LH.smallerthanmean, df_results_LH.largerthanmean, s = 20, color = df_results_LH.color)
ax[0,0].plot( [0,1],[0,1] )
ax[0,0].set_ylabel('Temperatures above mean')
ax[0,0].set_title ('LH')
red_patch = mpatches.Circle((0, 0), color='Firebrick', label= 'mean T > 22 $^\circ$C')
blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label= 'mean T < 2 $^\circ$C')
ax[0,0].text(-0.1,1.1, "(a)", size = 10,  horizontalalignment='center', 
                    verticalalignment='center', transform=ax[0,0].transAxes)

ax[0,0].legend(handles=[blue_patch, red_patch], loc = 4)
ax[1,0].scatter(df_results_SH.smallerthanmean, df_results_SH.largerthanmean, s =20, color = df_results_SH.color)
ax[1,0].plot( [0,1],[0,1] )
ax[1,0].set_ylabel('Temperatures above mean')
ax[1,0].set_title('SH')

ax[2,0].scatter(df_results_NEE.smallerthanmean, df_results_NEE.largerthanmean, s= 20, color = df_results_NEE.color)
ax[2,0].plot( [0,1],[0,1] )
ax[2,0].set_xlabel('Temperatures below mean')
ax[2,0].set_ylabel('Temperatures above mean')
ax[2,0].set_title('NEE ') 


ax[0,1].scatter(df_results_LH.smallerthan2stdev, df_results_LH.largerthan2stdev, s=20, color = df_results_LH.color)
ax[0,1].plot( [0,1],[0,1] )
ax[0,1].set_ylabel('Upper extreme')
ax[0,1].set_title ('LH ')
ax[0,1].text(-0.1,1.1, "(b)", size = 10,  horizontalalignment='center', 
                    verticalalignment='center', transform=ax[0,1].transAxes)

ax[1,1].scatter(df_results_SH.smallerthan2stdev, df_results_SH.largerthan2stdev, s = 20, color = df_results_SH.color)
ax[1,1].plot( [0,1],[0,1] )
ax[1,1].set_ylabel('Upper extreme')
ax[1,1].set_title('SH ')

ax[2,1].scatter(df_results_NEE.smallerthan2stdev, df_results_NEE.largerthan2stdev, s=20, color = df_results_NEE.color)
ax[2,1].plot( [0,1],[0,1] )
ax[2,1].set_xlabel('Lower extreme')
ax[2,1].set_ylabel('Upper extreme')
ax[2,1].set_title('NEE ') 

plot_dir = "../plots"
ofname = "scatterplots.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)
