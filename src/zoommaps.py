#!/usr/bin/env python

"""
I use this script to make zoommaps which show the ratio of measurements of
Qle, Qh and NEE for each FLUXNET and LaThuille site. 

"""

__author__ = "Sophie van der Horst"
__version__ = "1.0 (25.10.2018)"
__email__ = "sophie.vanderhorst@wur.nl"


import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
import matplotlib.backends.backend_pdf #import PdfPages
import matplotlib
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

lons = df_results_LH['lon'].tolist()
lats = df_results_LH['lat'].tolist()


def zoommap(vals, title, label):
    fig =  plt.figure(figsize=(12,8))    
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,
                        wspace=0.15,hspace=0.05)
    ax = plt.subplot(211)
    m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
                urcrnrlat= 82, urcrnrlon = 170, resolution = 'c')
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth = 0.5)
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,
                        wspace=0.15,hspace=0.05)
    cmap = matplotlib.colors.ListedColormap(['#d73027', '#fc8d59', '#fee090', 
                                             '#e0f3f8', '#91bfdb', '#4575b4'])
    bounds = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    m.scatter(x, y, s = 40, c = vals, cmap=cmap, norm=norm)
    plt.colorbar(orientation="vertical", boundaries = bounds, 
                 spacing = 'proportional', 
                 ticks = bounds)
    ax.text(0.03,0.95, label, size = 12,  horizontalalignment='center', 
    verticalalignment='center', transform=ax.transAxes)
    plt.title(title, fontsize = 12)
    #Zoom Europe
    axins_1 = zoomed_inset_axes(ax, 2, loc=2, bbox_to_anchor=(0.396, 0.48),
                         bbox_transform=ax.figure.transFigure)
    axins_1.scatter(x, y, s = 20, c = vals, cmap=cmap,norm=norm)
    m.drawcoastlines(linewidth = 0.5)
    x2,y2 = m(-12,35) 
    x3,y3 = m(40,65)
    axins_1.set_xlim(x2,x3) 
    axins_1.set_ylim(y2,y3) 
    axes = mark_inset(ax, axins_1, loc1=1, loc2=2, linewidth=1)
    #Zoom Australia
    axins_2 = zoomed_inset_axes(ax, 2.2, loc=3, bbox_to_anchor=(0.59, 0.255),
                         bbox_transform=ax.figure.transFigure)
    axins_2.scatter(x, y, s = 20, c = vals, cmap=cmap, norm=norm)
    m.drawcoastlines(linewidth = 0.5)
    x2,y2 = m(110,-43) 
    x3,y3 = m(155,-10)
    axins_2.set_xlim(x2,x3) 
    axins_2.set_ylim(y2,y3) 
    axes = mark_inset(ax, axins_2, loc1=1, loc2=2,linewidth=1)
    #Zoom US
    axins_3 = zoomed_inset_axes(ax, 1.6, loc=3, bbox_to_anchor=(0.19, 0.25),
                         bbox_transform=ax.figure.transFigure)
    axins_3.scatter(x, y, s = 20, c = vals, cmap=cmap, norm=norm)
    m.drawcoastlines(linewidth = 0.5)
    x2,y2 = m(-130,22) 
    x3,y3 = m(-60,63)
    axins_3.set_xlim(x2,x3) 
    axins_3.set_ylim(y2,y3) 
    axes = mark_inset(ax, axins_3, loc1=1, loc2=2, linewidth=1)
    return(fig, axes)


# creating baseplots with zoom  
#overall performance LH, SH, NEE

p1 = zoommap(df_results_LH.overall, 
             "Overall Qle measurement ratio", "(a)")
p2 = zoommap(df_results_LH.smallerthan2stdev, 
             "Qle measurement ratio for the lower 2.2 % temperatures", "(b)")
p3 = zoommap(df_results_LH.largerthan2stdev,
             "Qle measurement ratio for the upper 2.2 % temperatures","(c)")
p4 = zoommap(df_results_SH.overall, 
             "Overall Qh measurement ratio", "(a)")
p5 = zoommap(df_results_SH.smallerthan2stdev, 
             "Qh measurement ratio for the lower 2.2 % temperatures", "(b)")
p6 = zoommap(df_results_SH.largerthan2stdev, 
             "Qh measurement ratio for the upper 2.2 % temperatures", "(c)")
p7 = zoommap(df_results_NEE.overall, 
             "Overall NEE measurement ratio", "(a)")
p8 = zoommap(df_results_NEE.smallerthan2stdev, 
             "NEE measurement ratio for the lower 2.2 % temperatures", "(b)")
p9 = zoommap(df_results_NEE.largerthan2stdev, 
             "NEE measurement ratio for the upper 2.2 % temperatures", "(c)")


plot_dir = "../plots/Zoommaps"
p1[0].savefig(os.path.join(plot_dir, "Qle overall.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p2[0].savefig(os.path.join(plot_dir, "Qle lower.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p3[0].savefig(os.path.join(plot_dir, "Qle upper.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p4[0].savefig(os.path.join(plot_dir, "Qh overall.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p5[0].savefig(os.path.join(plot_dir, "Qh lower.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p6[0].savefig(os.path.join(plot_dir, "Qh upper.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p7[0].savefig(os.path.join(plot_dir, "NEE overall.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p8[0].savefig(os.path.join(plot_dir, "NEE lower.pdf"),
            bbox_inches='tight', pad_inches=0.1)
p9[0].savefig(os.path.join(plot_dir, "NEE upper.pdf"),
            bbox_inches='tight', pad_inches=0.1)














