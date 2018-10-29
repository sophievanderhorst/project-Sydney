#!/usr/bin/env python

"""
This is used to select sites with the highest measurement ratios of Qle, Qh 
and NEE. Sites are selected where all ratios are above 0.9 or 0.8. Also, sites are
selected where Qle and Qh are above 0.8 or 0.9.
"""
    # Import packages
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import matplotlib.patches as mpatches
import os

__author__ = "Sophie van der Horst"
__version__ = "1.0 (25.10.2018)"
__email__ = "sophie.vanderhorst@wur.nl"

with open("../data/processed/results_LH.csv", newline='') as myFile:  
    df_results_LH = pd.read_csv(myFile)
    for row in df_results_LH:
        print(row) 
        
with open("../data/processed/results_SH.csv", newline='') as myFile:  
    df_results_SH = pd.read_csv(myFile)
    for row in df_results_SH:
        print(row) 

with open("../data/processed/results_NEE.csv", newline='') as myFile:  
    df_results_NEE = pd.read_csv(myFile)
    for row in df_results_NEE:
        print(row) 

#Renaming and putting DataFrames together
df_results_LH1 = df_results_LH.rename (columns={'overall': 'overallLH', 
                                 'smallerthan2stdev': 'smallerthan2stdevLH', 
                                 'largerthan2stdev': 'largerthan2stdevLH'})
df_results_SH1 = df_results_SH.rename (columns={'overall': 'overallSH', 
                                 'smallerthan2stdev': 'smallerthan2stdevSH', 
                                 'largerthan2stdev': 'largerthan2stdevSH'})
df_results_NEE1 = df_results_NEE.rename (columns={'overall': 'overallNEE', 
                                 'smallerthan2stdev': 'smallerthan2stdevNEE', 
                                 'largerthan2stdev': 'largerthan2stdevNEE'})    
df_results = result = pd.concat([df_results_LH1.site, df_results_LH1.overallLH, df_results_LH1.smallerthan2stdevLH,
                                 df_results_LH1.largerthan2stdevLH,df_results_SH1.overallSH, 
                                 df_results_SH1.smallerthan2stdevSH,df_results_SH1.largerthan2stdevSH,
                                 df_results_NEE1.overallNEE, df_results_NEE1.smallerthan2stdevNEE,
                                 df_results_NEE1.largerthan2stdevNEE, df_results_LH1.lon, 
                                 df_results_LH1.lat], axis = 1)

#Overall measurement ratios for all above 0.9, all above 0.8, SH and LH above 0.9 and SH and LH above 0.8
overallbestsitesall = df_results[(df_results['overallLH'] >= 0.9) & 
                                 (df_results['overallSH'] >= 0.9) &
                                 (df_results['overallNEE'] >= 0.9)]
overall2ndbestsitesall = df_results [(df_results['overallLH'] >= 0.8)& 
                                     (df_results['overallSH'] >= 0.8) &
                                     (df_results['overallNEE'] >= 0.8)] 
overallbestsitesSHLH = df_results [(df_results['overallLH'] >= 0.9) & 
                                   (df_results['overallSH'] >= 0.9) &
                                    (df_results['overallNEE']< 0.8)]
overall2ndbestsitesSHLH = df_results [(df_results['overallLH'] >= 0.8) &
                                      (df_results['overallSH'] >= 0.8) & 
                                      (df_results['overallNEE'] < 0.8)]

# all above 90%
overallbestsitesall = pd.concat([overallbestsitesall.site, overallbestsitesall.overallLH, 
                      overallbestsitesall.overallSH, overallbestsitesall.overallNEE, 
                      overallbestsitesall.lon, overallbestsitesall.lat], axis =1)
overallbestsitesall['color'] = '#4575b4'
# all above 80%
overall2ndbestsitesall = pd.concat([overall2ndbestsitesall.site, overall2ndbestsitesall.overallLH, 
                      overall2ndbestsitesall.overallSH, overall2ndbestsitesall.overallNEE,
                      overall2ndbestsitesall.lon, overall2ndbestsitesall.lat], axis =1)
overall2ndbestsitesall['color'] = 'red'
#SH and LH above 90%
overallbestsitesSHLH = pd.concat([overallbestsitesSHLH.site, overallbestsitesSHLH.overallLH, 
                      overallbestsitesSHLH.overallSH, overallbestsitesSHLH.overallNEE,
                      overallbestsitesSHLH.lon, overallbestsitesSHLH.lat], axis =1)
overallbestsitesSHLH['color'] = 'orange'
#SH and LH above 80%
overall2ndbestsitesSHLH = pd.concat([overall2ndbestsitesSHLH.site, overall2ndbestsitesSHLH.overallLH, 
                      overall2ndbestsitesSHLH.overallSH, overall2ndbestsitesSHLH.overallNEE,
                      overall2ndbestsitesSHLH.lon, overall2ndbestsitesSHLH.lat], axis =1)
overall2ndbestsitesSHLH['color'] = 'purple'

#putting results together and tropping duplicates
table_overall = overallbestsitesall.append([overall2ndbestsitesall, 
                                            overallbestsitesSHLH, overall2ndbestsitesSHLH ])
table_overall = table_overall.drop_duplicates(subset=['site', 'overallLH', 'overallSH', 'overallNEE'],
                                              keep='first', inplace=False)

#Table creating for overall measurement ratio
fig, ax = plt.subplots()
plt.title("Overall", y = 2.2)
ax.axis('off')
colours = table_overall.color.values
table_overall_stripped = table_overall[['site','overallLH', 'overallSH', 'overallNEE']]
table_overall_stripped = table_overall_stripped.set_index('site')
ax.table(cellText=table_overall_stripped.values, 
         rowColours = colours,
         colLabels=("LH", "SH", "NEE"), 
         rowLabels = table_overall_stripped.index.values,
         loc='center')
blue_patch = mpatches.Patch(color='#4575b4', label='LH, SH and NEE above 0.9')
red_patch = mpatches.Patch(color='red', label='LH, SH and NEE above 0.8')
orange_patch = mpatches.Patch(color='orange', label='LH, SH above 0.9')
purple_patch = mpatches.Patch(color='purple', label='LH, SH above 0.8')
ax.legend(handles=[blue_patch, red_patch, orange_patch, purple_patch], bbox_to_anchor = (1.25, 1.9, 0.3, 0.3))

plot_dir = "../plots"
ofname = "table_suitable_overall.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)


#Lower extreme measurement ratios for all above 0.9, all above 0.8, SH and LH above 0.9 and SH and LH above 0.8
lowerbestsitesall = df_results[(df_results['smallerthan2stdevLH'] >= 0.9) & 
                                 (df_results['smallerthan2stdevSH'] >= 0.9) &
                                 (df_results['smallerthan2stdevNEE'] >= 0.9)]
lower2ndbestsitesall = df_results [(df_results['smallerthan2stdevLH'] >= 0.8)& 
                                     (df_results['smallerthan2stdevSH'] >= 0.8) &
                                     (df_results['smallerthan2stdevNEE'] >= 0.8)] 
lowerbestsitesSHLH = df_results [(df_results['smallerthan2stdevLH'] >= 0.9) & 
                                   (df_results['smallerthan2stdevSH'] >= 0.9) &
                                    (df_results['smallerthan2stdevNEE']< 0.8)]
lower2ndbestsitesSHLH = df_results [(df_results['smallerthan2stdevLH'] >= 0.8) &
                                      (df_results['smallerthan2stdevSH'] >= 0.8) & 
                                      (df_results['smallerthan2stdevNEE'] < 0.8)]

# all above 90%
lowerbestsitesall = pd.concat([lowerbestsitesall.site, lowerbestsitesall.smallerthan2stdevLH, 
                      lowerbestsitesall.smallerthan2stdevSH, lowerbestsitesall.smallerthan2stdevNEE, 
                      lowerbestsitesall.lon, lowerbestsitesall.lat], axis =1)
lowerbestsitesall['color'] = '#4575b4'
# all above 80%
lower2ndbestsitesall = pd.concat([lower2ndbestsitesall.site, lower2ndbestsitesall.smallerthan2stdevLH, 
                      lower2ndbestsitesall.smallerthan2stdevSH, lower2ndbestsitesall.smallerthan2stdevNEE,
                      lower2ndbestsitesall.lon, lower2ndbestsitesall.lat], axis =1)
lower2ndbestsitesall['color'] = 'red'
#SH and LH above 90%
lowerbestsitesSHLH = pd.concat([lowerbestsitesSHLH.site, lowerbestsitesSHLH.smallerthan2stdevLH, 
                      lowerbestsitesSHLH.smallerthan2stdevSH, lowerbestsitesSHLH.smallerthan2stdevNEE,
                      lowerbestsitesSHLH.lon, lowerbestsitesSHLH.lat], axis =1)
lowerbestsitesSHLH['color'] = 'orange'
#SH and LH above 80%
lower2ndbestsitesSHLH = pd.concat([lower2ndbestsitesSHLH.site, lower2ndbestsitesSHLH.smallerthan2stdevLH, 
                      lower2ndbestsitesSHLH.smallerthan2stdevSH, lower2ndbestsitesSHLH.smallerthan2stdevNEE,
                      lower2ndbestsitesSHLH.lon, lower2ndbestsitesSHLH.lat], axis =1)
lower2ndbestsitesSHLH['color'] = 'purple'
#putting results together and tropping duplicates
table_lower= lowerbestsitesall.append([lower2ndbestsitesall, lowerbestsitesSHLH, 
                                       lower2ndbestsitesSHLH ])
table_lower = table_lower.drop_duplicates(subset=['site', 'smallerthan2stdevLH',
                                                  'smallerthan2stdevSH', 'smallerthan2stdevNEE'],
                                          keep='first', inplace=False)

#Table creation for Lower extreme
fig, ax = plt.subplots()
plt.title("Lower extreme", y = 1.5)
ax.axis('off')
colours = table_lower.color.values
table_lower_stripped = table_lower[['site', 'smallerthan2stdevLH',
                                    'smallerthan2stdevSH', 'smallerthan2stdevNEE']]
table_lower_stripped = table_lower_stripped.set_index('site')
ax.table(cellText=table_lower_stripped.values,
         rowColours = colours, 
         colLabels=("LH", "SH", "NEE"), 
         rowLabels = table_lower_stripped.index.values,
         loc='center')
ax.legend(handles=[blue_patch, red_patch, orange_patch, purple_patch],
          bbox_to_anchor = (1.25, 1.2, 0.3, 0.3))

ofname = "table_suitable_lowerextreme.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)

#Upper extreme measurement ratios for all above 0.9, all above 0.8, SH and LH above 0.9 and SH and LH above 0.8
upperbestsitesall = df_results[(df_results['largerthan2stdevLH'] >= 0.9) & 
                                 (df_results['largerthan2stdevSH'] >= 0.9) &
                                 (df_results['largerthan2stdevNEE'] >= 0.9)]
upper2ndbestsitesall = df_results [(df_results['largerthan2stdevLH'] >= 0.8)& 
                                     (df_results['largerthan2stdevSH'] >= 0.8) &
                                     (df_results['largerthan2stdevNEE'] >= 0.8)] 
upperbestsitesSHLH = df_results [(df_results['largerthan2stdevLH'] >= 0.9) & 
                                   (df_results['largerthan2stdevSH'] >= 0.9) &
                                    (df_results['largerthan2stdevNEE']< 0.8)]
upper2ndbestsitesSHLH = df_results [(df_results['largerthan2stdevLH'] >= 0.8) &
                                      (df_results['largerthan2stdevSH'] >= 0.8) & 
                                      (df_results['largerthan2stdevNEE'] < 0.8)]

# all above 90%
upperbestsitesall = pd.concat([upperbestsitesall.site, upperbestsitesall.largerthan2stdevLH, 
                      upperbestsitesall.largerthan2stdevSH, upperbestsitesall.largerthan2stdevNEE,
                      upperbestsitesall.lon, upperbestsitesall.lat], axis =1)
upperbestsitesall['color'] = '#4575b4'
# all above 80%
upper2ndbestsitesall = pd.concat([upper2ndbestsitesall.site, upper2ndbestsitesall.largerthan2stdevLH, 
                      upper2ndbestsitesall.largerthan2stdevSH, upper2ndbestsitesall.largerthan2stdevNEE,
                      upper2ndbestsitesall.lon, upper2ndbestsitesall.lat], axis =1)
upper2ndbestsitesall['color'] = 'red'
#SH and LH above 90%
upperbestsitesSHLH = pd.concat([upperbestsitesSHLH.site, upperbestsitesSHLH.largerthan2stdevLH, 
                      upperbestsitesSHLH.largerthan2stdevSH, upperbestsitesSHLH.largerthan2stdevNEE,
                      upperbestsitesSHLH.lon, upperbestsitesSHLH.lat], axis =1)
upperbestsitesSHLH['color'] = 'orange'
#SH and LH above 80%
upper2ndbestsitesSHLH = pd.concat([upper2ndbestsitesSHLH.site, upper2ndbestsitesSHLH.largerthan2stdevLH, 
                      upper2ndbestsitesSHLH.largerthan2stdevSH, upper2ndbestsitesSHLH.largerthan2stdevNEE,
                      upper2ndbestsitesSHLH.lon, upper2ndbestsitesSHLH.lat], axis =1)
upper2ndbestsitesSHLH['color'] = 'purple'
#putting results together and tropping duplicates
table_upper= upperbestsitesall.append([upper2ndbestsitesall, upperbestsitesSHLH, 
                                       upper2ndbestsitesSHLH ])
table_upper = table_upper.drop_duplicates(subset=['site', 'largerthan2stdevLH',
                                                 'largerthan2stdevSH', 'largerthan2stdevNEE'], 
                                          keep='first', inplace=False)

#Table creation for upper extreme
fig, ax = plt.subplots()
plt.title ("Upper extreme", y = 2.95)
ax.axis('off')
colours = table_upper.color.values
table_upper_stripped = table_upper[['site', 'largerthan2stdevLH',
                                    'largerthan2stdevSH', 'largerthan2stdevNEE']]
table_upper_stripped = table_upper_stripped.set_index('site')
ax.table(cellText=table_upper_stripped.values,
         rowColours = colours, 
         colLabels=("LH", "SH", "NEE"), 
         rowLabels = table_upper_stripped.index.values, 
         loc='center')
ax.legend(handles=[blue_patch, red_patch, orange_patch, purple_patch], 
          bbox_to_anchor = (1.25, 2.6, 0.3, 0.3))


ofname = "table_suitable_upperextreme.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)

#Show the results of the table in basemaps
#Baseplot for overall performance
fig =  plt.figure(figsize=(11,8))
plt.title("Overall suitable sites", fontsize = 12)   
plt.text(0.04,1.3, "(a)", size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)
plt.legend(handles=[blue_patch, red_patch, orange_patch, purple_patch], loc = 3) 
m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
            urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
m.drawcoastlines(linewidth = 0.5)
for row in table_overall.itertuples():
    x,y = m(table_overall.lon.values, table_overall.lat.values)
    m.scatter(x,y, s = 20, color = table_overall.color.values)
    

ofname = "map_suitable_overall.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)
    
#Baseplot for lower extreme performance
fig =  plt.figure(figsize=(11,8))
plt.title("Lower extreme suitable sites", fontsize = 12)  
plt.text(0.04,1.3, "(b)", size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)
m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
            urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
m.drawcoastlines(linewidth = 0.5)
for row in table_lower.itertuples():
    x,y = m(table_lower.lon.values, table_lower.lat.values)
    m.scatter(x,y, s = 20, color = table_lower.color.values)

ofname = "map_suitable_lowerextreme.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)
    
#Baseplot for upper extreme performance
fig =  plt.figure(figsize=(11,8))
plt.title("Upper extreme suitable sites", fontsize = 12) 
plt.text(0.04,1.3, "(c)", size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)  
m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
            urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
m.drawcoastlines(linewidth = 0.5)
for row in table_upper.itertuples():
    x,y = m(table_upper.lon.values, table_upper.lat.values)
    m.scatter(x,y, s = 20, color = table_upper.color.values)
    
ofname = "map_suitable_upperextreme.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)
