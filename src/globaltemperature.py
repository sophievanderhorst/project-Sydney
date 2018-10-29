#!/usr/bin/env python

"""
I use this script to make plots that show the number of measurements 
and measurements for all measured temperatures globally. 

"""

__author__ = "Sophie van der Horst"
__version__ = "1.0 (25.10.2018)"
__email__ = "sophie.vanderhorst@wur.nl"

import matplotlib.pyplot as plt
import pandas as pd
import os 
 
with open("../data/processed/globaltemp.csv", newline='') as myFile:  
    df_total = pd.read_csv(myFile)
    for row in df_total:
        print(row)          

df_total = df_total.set_index("tempbin")
x_axis = df_total.index - 273

#barplot with absolute numbers
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (12,5))
ax[0].plot(x_axis, df_total['Temp nor'], color = '#0571b0') 
ax[0].plot(x_axis, df_total['Qle nor'], color = '#92c5de' )
ax[0].plot(x_axis, df_total['Qh nor'], color = '#ca0020') 
ax[0].plot(x_axis, df_total['NEE nor'], color = '#f4a582')
ax[0].set_title("Number of measured temperatures, Qh, Qle, NEE (globally)", fontsize = 12)
ax[0].set_xlabel('Temperature ($^\circ$C)', fontsize = 12)
ax[0].set_ylabel('Measurements (normalised)', fontsize = 12) 
ax[0].legend(labels = ["Temperature", "Qle", "Qh", "NEE"])
ax[0].text(-0.1,1.1, "(a)", size = 12,  horizontalalignment='center', 
  verticalalignment='center', transform=ax[0].transAxes) 
ax[1].plot(x_axis, df_total['ratio Qle'], color = '#92c5de') 
ax[1].plot(x_axis, df_total['ratio Qh'], color = '#ca0020')
ax[1].plot(x_axis, df_total['ratio NEE'], color = '#f4a582' ) 
ax[1].set_title("Ratio of measured Qle, Qh, NEE (globally)", fontsize = 12) 
ax[1].set_xlabel('Temperature ($^\circ$C)', fontsize = 12)
ax[1].set_ylabel('Measurement ratio', fontsize = 12)
ax[1].text(-0.1,1.1, "(b)", size = 12,  horizontalalignment='center', 
  verticalalignment='center', transform=ax[1].transAxes)
plt.tight_layout()

plot_dir = "../plots"
ofname = "global_temp.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)