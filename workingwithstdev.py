#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:02:14 2018

@author: z5228341
"""
    # Import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from mpl_toolkits.basemap import Basemap
from scipy.stats import norm
import pandas as pd
import glob
import xarray as xr
import plotly.plotly as py
import plotly.graph_objs as go
from mpl_toolkits.axes_grid.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid.inset_locator import mark_inset
import matplotlib.backends.backend_pdf #import PdfPages
import os
import matplotlib
import matplotlib.patches as mpatches
import plotly
import scipy
from matplotlib.colors import ListedColormap


#Import the files
#files_met = sorted(glob.glob("/home/z5228341/Desktop/MET/*"))
files_met = sorted(glob.glob("/home/z5228341/Desktop/FLUXNET & LaThuille/MET/Met/*"))

#files_flux = sorted(glob.glob("/home/z5228341/Desktop/FLUX/*"))
files_flux = sorted(glob.glob("/home/z5228341/Desktop/FLUXNET & LaThuille/FLUX/Flux/*"))

#Setting up figures to which I can add subplots
fig0 = plt.figure()
fig1 = plt.figure(figsize = (12,18))
fig2 = plt.figure(figsize = (12,18))
fig3 = plt.figure(figsize = (12,27))
fig4 = plt.figure(figsize = (12,18))
fig5 = plt.figure (figsize = (12,18))
fig6 = plt.figure(figsize = (12,27))
fig7 = plt.figure(figsize = (12,18))
fig8 = plt.figure (figsize = (12,18))
fig9 = plt.figure(figsize = (12,27))
fig10 = plt.figure()
fig11 = plt.figure()
fig12 = plt.figure()
fig13 = plt.figure(figsize = (10,15))
fig15 = plt.figure(figsize = (12,18))

#function to open and organize data
def open_file(fname):
    ds = xr.open_dataset(fname)
    ds = ds.squeeze(dim=["x","y"], drop=True).to_dataframe()
    ds = ds.reset_index()
    ds = ds.set_index('time') 
    return (ds)
    
#fuction for making temperature-precipitation plots
def prectemp(temp, precip, vals, title, ax, label):
    cmap = matplotlib.colors.ListedColormap(['#d73027', '#fc8d59', '#fee090', 
                                             '#e0f3f8', '#91bfdb', '#4575b4'])
    norm = matplotlib.colors.BoundaryNorm([0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0], cmap.N)
    sc = ax.scatter(temp, precip, s= 30, c= vals, cmap = cmap,
                            norm = norm  )
    plt.colorbar(sc, ax = ax, orientation="vertical", 
                 boundaries = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0], 
                 spacing = 'proportional', 
                 ticks = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0])
    
    ax.set_xlabel("Temperature (C)")
    ax.set_ylabel("Precipitation (m)")
    ax.set_title(title)
    ax.text(-0.1,1.1, label, size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)
    return(fig) 
   
    
#fuction for making maps with zooms for Europe, North-America and Australia
def zoommap(vals, title, label):
    fig =  plt.figure(figsize=(12,8))    
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,
                        wspace=0.15,hspace=0.05)
    ax = plt.subplot(211)
    m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
                urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
    x,y = m(lons, lats)
    m.drawcoastlines()
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
    ax.text(-0.1,1.1, label, size = 12,  horizontalalignment='center', 
    verticalalignment='center', transform=ax.transAxes)
    plt.title(title)
    #Zoom Europe
    axins_1 = zoomed_inset_axes(ax, 2, loc=2, bbox_to_anchor=(0.396, 0.48),
                         bbox_transform=ax.figure.transFigure)
    axins_1.scatter(x, y, s = 20, c = vals, cmap=cmap,norm=norm)
    m.drawcoastlines()
    x2,y2 = m(-12,35) 
    x3,y3 = m(40,65)
    axins_1.set_xlim(x2,x3) 
    axins_1.set_ylim(y2,y3) 
    axes = mark_inset(ax, axins_1, loc1=1, loc2=2, linewidth=1)
    #Zoom Australia
    axins_2 = zoomed_inset_axes(ax, 2.2, loc=3, bbox_to_anchor=(0.59, 0.255),
                         bbox_transform=ax.figure.transFigure)
    axins_2.scatter(x, y, s = 20, c = vals, cmap=cmap, norm=norm)
    m.drawcoastlines()
    x2,y2 = m(110,-43) 
    x3,y3 = m(155,-10)
    axins_2.set_xlim(x2,x3) 
    axins_2.set_ylim(y2,y3) 
    axes = mark_inset(ax, axins_2, loc1=1, loc2=2,linewidth=1)
    #Zoom US
    axins_3 = zoomed_inset_axes(ax, 1.6, loc=3, bbox_to_anchor=(0.19, 0.25),
                         bbox_transform=ax.figure.transFigure)
    axins_3.scatter(x, y, s = 20, c = vals, cmap=cmap, norm=norm)
    m.drawcoastlines()
    x2,y2 = m(-130,22) 
    x3,y3 = m(-60,63)
    axins_3.set_xlim(x2,x3) 
    axins_3.set_ylim(y2,y3) 
    axes = mark_inset(ax, axins_3, loc1=1, loc2=2, linewidth=1)
    return(fig, axes)

#Function for heatmap (table-like)    
def heatmap(ax, data, sites, label, cmap, norm):
    im = ax.imshow(data, cmap = cmap, norm = norm, interpolation = 'nearest', 
                   aspect = 'auto')
    font = {'family': 'Arial Narrow', 'color':  'black','weight': 'normal', 
            'size': 8,}
    ax.set_xticklabels(label, fontdict = font)
    ax.set_yticklabels(sites, fontdict = font)
    ax.set_xticks(np.arange(len(label)))
    ax.set_yticks(np.arange(len(sites)))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    return im

#Function to format heatmaps (make certain text white and don't show the 0)   
def annotate_heatmap(im, data = None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=0.9, **textkw):    
    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()
    kw = dict(horizontalalignment="center", verticalalignment="center")
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j]>=0 or data[i,j]<=0:
                kw.update(color=textcolors[(data[i, j]) >= threshold])
                kw.update (size = 8)
                kw.update (family = 'Arial Narrow')
                text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
                texts.append(text)
    return texts

#Function for formatting the tables 
def func(x, pos):
    return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "1.0")


# empty lists to add data in for the table/figures
resultsLH = []
resultsSH = []
resultsNEE = []
lons=[]
lats=[]

#creating dataframes for barplots of global temperature distribution
df_temp = pd.DataFrame()
df_LH = pd.DataFrame()
df_SH = pd.DataFrame()
df_NEE = pd.DataFrame()

#creation of for loop which analyzes each Fluxdataset
for m,f in zip(files_met, files_flux):
    print(m,f)
      
    #Opening the files 
    ds_met = open_file(m)
    ds_flux = open_file(f)

    #selecting on lon and lat
    lat = ds_met['latitude'].mean()
    lon = ds_met['longitude'].mean() 
    
    #Does the data have enough recorded temperatures? First, selecting 
    #only measurements with qc = 0. Second, two conditions: the total length
    #of the dataset and the percentage of measured temperatures. 
    Tair_measured = ds_met[ds_met.Tair_qc < 1]
    mean_tair_measured = np.mean(Tair_measured.Tair)
     #not all datasets had a qc for this. 
    if len(ds_met) > ((2*365*48)/3) and (len(Tair_measured)/len(ds_met))>0.5:
          
            # Filter to measured temperature (qc=0) and only shortwave 
            #incoming radiation >0 and filtering out data between 11 pm and 6 am
            ds_flux = ds_flux[ds_met.Tair_qc < 1]
            ds_met = ds_met[ds_met.Tair_qc < 1]
            ds_flux = ds_flux[ds_met.SWdown > 1]
            ds_met = ds_met[ds_met.SWdown > 1]
            ds_flux = ds_flux.drop(ds_flux.between_time("23:00", "6:00").index)
            ds_met = ds_met.drop(ds_met.between_time("23:00", "6:00").index)
            Mean_precipitation_measured_yearly = ((ds_met.Rainf.mean())*48*365)
            
            

            # Create plot normal distribution of the temperature
            stdev_tair_measured = np.std(Tair_measured.Tair)
            parameters = norm.fit(ds_met.Tair)
            x = np.linspace(min(ds_met.Tair), max(ds_met.Tair), 1000)
            normal_pdf = norm.pdf(x, mean_tair_measured, stdev_tair_measured)      
            ax = fig0.add_subplot(111)
            ax.plot(x, normal_pdf, "blue", label=m, linewidth=2)
            ax.set_title("Temperature distribution") 
            ax.set_xlabel("Temperature")                                      
            ax.set_ylabel("Frequency") 
     
            #creating bins of 1 K and assigning a temperature to a bin. 
            
            minimum_tair = (min(ds_met.Tair))
            maximum_tair = (max(ds_met.Tair))
            bins = np.arange(np.floor(minimum_tair), 
                             np.ceil(maximum_tair+1)).astype(int)
            bin_label = np.arange(np.floor(minimum_tair), 
                                  np.ceil(maximum_tair)).astype(int)
            data_binned = pd.cut(ds_met.Tair, bins, labels=bin_label)
            #Adding this temperaturebin to the datasets.
            ds_met.loc[:,"tempbin"] = data_binned
            ds_flux.loc[:, "tempbin"] = data_binned
            
            #For each bin, count the amount of measurements create a 
            #pandas Dataframe for temperature
            temp_sorted = ds_met.groupby(['tempbin']).size()
            #Create a pandas Dataframe for temperature and add binlabel
            dftemp_sorted= pd.DataFrame(temp_sorted)
            dftemp_sorted.loc[:,"bin_label"]=bin_label
            
            #LATENT HEAT
            #filtering out only the measured latent heats
            ds_met_measuredLH = ds_met[ds_flux.Qle_qc<1]
            ds_flux_measuredLH = ds_flux[ds_flux.Qle_qc<1]
            
            #Ordering the data according to the temperaturebin, making a
            # pandaframe and equalizing the index to the index
            #of the temperature dataframe. 
            ds_flux_measuredLH_sorted = ds_flux_measuredLH.groupby(['tempbin']).size()
            df_flux_measuredLH_sorted = pd.DataFrame(ds_flux_measuredLH_sorted)
            dftemp_sorted = dftemp_sorted.set_index(df_flux_measuredLH_sorted.index)
            measuredLH = df_flux_measuredLH_sorted.iloc[:,0]
            #adding the measured LH to temperature pandas dataframe
            dftemp_sorted.loc[:,"measuredLH"]=measuredLH            
            #for each temperature bin add: the fraction of LH measurements, the 
            #average temperature of the site, stdev 
            #of the temperature of the site and the stdev scale. 
            dftemp_sorted.loc[:,'ratioLH'] = dftemp_sorted['measuredLH']/\
                                                dftemp_sorted.iloc[:,0]
            dftemp_sorted.loc[:,"averagetemp"] = mean_tair_measured
            dftemp_sorted.loc[:,"stdev"] = stdev_tair_measured
            dftemp_sorted.loc[:,"stdevscale"] = (((dftemp_sorted.loc[:,"bin_label"])-
                                                 (dftemp_sorted.loc[:,"averagetemp"]))/
                                                    (dftemp_sorted.loc[:,"stdev"]))

            #SENSIBLE HEAT
            #filtering out only the measured sensible heats
            ds_met_measuredSH = ds_met[ds_flux.Qh_qc<1]
            ds_flux_measuredSH = ds_flux[ds_flux.Qh_qc<1]

            #Ordering the data according to the temperaturebin, making a 
            #pandaframe and equalizing the index to the index of the temperature dataframe. 
            ds_flux_measuredSH_sorted = ds_flux_measuredSH.groupby(['tempbin']).size()
            df_flux_measuredSH_sorted = pd.DataFrame(ds_flux_measuredSH_sorted)
            dftemp_sorted = dftemp_sorted.set_index(df_flux_measuredSH_sorted.index)
            measuredSH=df_flux_measuredSH_sorted.iloc[:,0]
            #adding the measured SH to temperature pandas dataframe and the 
            #fraction of SH measurements
            dftemp_sorted.loc[:,"measuredSH"]=measuredSH
            dftemp_sorted.loc[:,'ratioSH'] = dftemp_sorted['measuredSH']/\
                                            dftemp_sorted.iloc[:,0]
            
            #NEE
            #filtering out only the measured NEE
            ds_met_measuredNEE = ds_met[ds_flux.NEE_qc<1]
            ds_flux_measuredNEE = ds_flux[ds_flux.NEE_qc<1]
            #Ordering the data according to the temperature bin, making a 
            #pandaframe and equalizing the index to the index of the temperature dataframe            
            ds_flux_measuredNEE_sorted = ds_flux_measuredNEE.groupby(['tempbin']).size()
            df_flux_measuredNEE_sorted = pd.DataFrame(ds_flux_measuredNEE_sorted)
            dftemp_sorted = dftemp_sorted.set_index(df_flux_measuredNEE_sorted.index)
            measuredNEE=df_flux_measuredNEE_sorted.iloc[:,0]
            #adding the measured NEE to temperature pandas dataframe and the 
            #fraction of NEE measurements
            dftemp_sorted.loc[:,"measuredNEE"]=measuredNEE
            dftemp_sorted.loc[:,'ratioNEE'] = dftemp_sorted['measuredNEE']/\
                                                dftemp_sorted.iloc[:,0]
            
            #only include temperature bins with >10 temperature measurements 
            dftemp_sorted = dftemp_sorted[dftemp_sorted[0]>10]
            
            #creating data for histogram for global temperature
            df_temp = pd.concat([df_temp, dftemp_sorted.iloc[:,0]], axis=1)
            df_LH = pd.concat([df_LH, dftemp_sorted['measuredLH']], axis=1)
            df_SH = pd.concat([df_SH, dftemp_sorted['measuredSH']], axis=1)
            df_NEE = pd.concat([df_NEE, dftemp_sorted['measuredNEE']], axis=1)
            
            # Working with the energy balance. Only measured SH, LH and Rnet. 
            # This was done over a long period of time. 
            ds_flux_energybalance = ds_flux[ds_flux.Qle_qc < 1]
            ds_flux_energybalance = ds_flux_energybalance[ds_flux.Qh_qc < 1]
            #not every dataset contained a Rnet.
            if 'Rnet' in ds_flux_energybalance.columns:
                ds_flux_energybalance_Rnet = ds_flux_energybalance[ds_flux.Rnet_qc <1]
                mean_Rnet = ds_flux_energybalance_Rnet.Rnet.mean()
                mean_LH = ds_flux_energybalance_Rnet.Qle.mean()
                mean_SH = ds_flux_energybalance_Rnet.Qh.mean()
                #Calculating the residual energy
                energybalance = mean_Rnet - mean_LH - mean_SH
            
            #dividing data in different temperature ranges for LH, SH and NEE 
            # average ratio (LH, SH, NEE measurements) for all temperatures
            ratioLH_overall_mean = (dftemp_sorted.measuredLH.sum()/
                                    dftemp_sorted[0].sum()).round(2)      
            ratioSH_overall_mean = (dftemp_sorted.measuredSH.sum()/
                                    dftemp_sorted[0].sum()).round(2)
            #not each dataset contains NEE
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_overall_mean = (dftemp_sorted.measuredNEE.sum()/
                                         dftemp_sorted[0].sum()).round(2)
            else: 
                ratioNEE_overall_mean = np.nan
            #This part is for temperature ranges I am not using in my study yet:
            #average ratio (LH, SH, NEE measurements) if temperature is < mean 
            smallerthan_mean = dftemp_sorted[dftemp_sorted.stdevscale<0]
            ratioLH_smallerthan_mean = (smallerthan_mean.measuredLH.sum()/
            smallerthan_mean[0].sum()).round(2) 
            ratioSH_smallerthan_mean = (smallerthan_mean.measuredSH.sum()/
            smallerthan_mean[0].sum()).round(2)
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_smallerthan_mean = (smallerthan_mean.measuredNEE.sum()/
                smallerthan_mean[0].sum()).round(2) 
            else:
                ratioNEE_smallerthan_mean = np.nan
            #average ratio (LH, SH, NEE measurements) if temperature is > mean 
            largerthan_mean = dftemp_sorted[dftemp_sorted.stdevscale>0]
            ratioLH_largerthan_mean = (largerthan_mean.measuredLH.sum()/
            largerthan_mean[0].sum()).round(2) 
            ratioSH_largerthan_mean = (largerthan_mean.measuredSH.sum()/
            largerthan_mean[0].sum()).round(2) 
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_largerthan_mean = (largerthan_mean.measuredNEE.sum()/
                largerthan_mean[0].sum()).round(2)
            else:
                ratioNEE_largerthan_mean = np.nan
            #average ratio (LH, SH, NEE measurements) if temperature withing 
            #1 stdev from the mean
            within1_stdev_mean = dftemp_sorted[(dftemp_sorted['stdevscale']<1) & 
                                             (dftemp_sorted['stdevscale'] >-1)]
            ratioLH_within1_stdev_mean = (within1_stdev_mean.measuredLH.sum()/
            within1_stdev_mean[0].sum()).round(2) 
            ratioSH_within1_stdev_mean = (within1_stdev_mean.measuredSH.sum()/
            within1_stdev_mean[0].sum()).round(2)
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_within1_stdev_mean = (within1_stdev_mean.measuredNEE.sum()/
                within1_stdev_mean[0].sum()).round(2)
            else:
                ratioNEE_within1_stdev_mean = np.nan               
            #average ratio (LH, SH, NEE measurements) if temperature is > 1stdev 
            #away from the mean
            largerthan1_stdev_mean = dftemp_sorted[(dftemp_sorted['stdevscale'] > 1)]
            ratioLH_largerthan1_stdev_mean = (largerthan1_stdev_mean.measuredLH.sum()/
            largerthan1_stdev_mean[0].sum()).round(2) 
            ratioSH_largerthan1_stdev_mean = (largerthan1_stdev_mean.measuredSH.sum()/
            largerthan1_stdev_mean[0].sum()).round(2) 
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_largerthan1_stdev_mean = (largerthan1_stdev_mean.measuredNEE.sum()/
                largerthan1_stdev_mean[0].sum()).round(2)
            else:
                ratioNEE_largerthan1_stdev_mean = np.nan
            #average ratio (LH, SH, NEE measurements) if temperature is < 1stdev 
            #away from the mean
            smallerthan1_stdev_mean = dftemp_sorted[(dftemp_sorted['stdevscale'] < -1)]
            ratioLH_smallerthan1_stdev_mean = (smallerthan1_stdev_mean.measuredLH.sum()/
            smallerthan1_stdev_mean[0].sum()).round(2) 
            ratioSH_smallerthan1_stdev_mean = (smallerthan1_stdev_mean.measuredSH.sum()/
            smallerthan1_stdev_mean[0].sum()).round(2) 
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_smallerthan1_stdev_mean = (smallerthan1_stdev_mean.measuredNEE.sum()/
                smallerthan1_stdev_mean[0].sum()).round(2)
            else:
                ratioNEE_smallerthan1_stdev_mean =np.nan
            #THESE I AM USING AGAIN
            #average ratio (LH, SH, NEE measurements) if temperature is > 2stdev 
            #away from the mean. Not all sites contain measurements in this bin. 
            #Therefore, this if-statement was created. 
            largerthan2_stdev_mean = dftemp_sorted[(dftemp_sorted['stdevscale'] > 2)]
            if len(largerthan2_stdev_mean) > 1:
                ratioLH_largerthan2_stdev_mean = (largerthan2_stdev_mean.measuredLH.sum()/
                                                  largerthan2_stdev_mean[0].sum()).round(2) 
                ratioSH_largerthan2_stdev_mean = (largerthan2_stdev_mean.measuredSH.sum()/
                                                  largerthan2_stdev_mean[0].sum()).round(2) 
                if dftemp_sorted.measuredNEE.sum()>1:
                    ratioNEE_largerthan2_stdev_mean = (largerthan2_stdev_mean.measuredNEE.sum()/
                                                       largerthan2_stdev_mean[0].sum()).round(2)
                else:
                    ratioNEE_largerthan2_stdev_mean = np.nan
            elif len(largerthan2_stdev_mean) == 0:
                ratioLH_largerthan2_stdev_mean = np.nan
                ratioSH_largerthan2_stdev_mean = np.nan
                ratioNEE_largerthan2_stdev_mean = np.nan
            #average ratio (LH, SH, NEE measurements) if temperature is < 2stdev 
            #away from the mean         
            smallerthan2_stdev_mean = dftemp_sorted[(dftemp_sorted['stdevscale'] < -2)]
            if len(smallerthan2_stdev_mean)>1:  
                ratioLH_smallerthan2_stdev_mean = (smallerthan2_stdev_mean.measuredLH.sum()/
                                                   smallerthan2_stdev_mean[0].sum()).round(2) 
                ratioSH_smallerthan2_stdev_mean = (smallerthan2_stdev_mean.measuredSH.sum()/
                                                   smallerthan2_stdev_mean[0].sum()).round(2) 
                if dftemp_sorted.measuredNEE.sum()>1:
                    ratioNEE_smallerthan2_stdev_mean = (smallerthan2_stdev_mean.measuredNEE.sum()/
                                                        smallerthan2_stdev_mean[0].sum()).round(2)
                else:
                    ratioNEE_smallerthan2_stdev_mean = np.nan
            elif len(smallerthan2_stdev_mean) == 0:
                ratioLH_smallerthan2_stdev_mean = np.nan
                ratioSH_smallerthan2_stdev_mean = np.nan
                ratioNEE_smallerthan2_stdev_mean = np.nan
                
            #shorten the filename to first 6 letters, which is the abberviation 
            #for the site
            mname = os.path.splitext(os.path.basename(m))[0]
            mname=mname[0:6]
            
            #vegetation
            vegetation = ds_met['IGBP_veg_long'].str.decode("utf-8")

            #colours for the scatterplot
            if mean_tair_measured >= 295:
                color1 = 'Firebrick'
            elif mean_tair_measured <275:
                color1 = 'blue'
            else:
                color1 = 'black'
            
            #Adding the results to the empty lists
            resultsLH.append([mname, mean_tair_measured,  Mean_precipitation_measured_yearly, 
                              ratioLH_overall_mean, ratioLH_smallerthan_mean, ratioLH_largerthan_mean,
                              ratioLH_within1_stdev_mean, ratioLH_smallerthan1_stdev_mean, 
                              ratioLH_largerthan1_stdev_mean, ratioLH_smallerthan2_stdev_mean,
                              ratioLH_largerthan2_stdev_mean, color1, energybalance, vegetation[0]]) 
            resultsSH.append([mname, mean_tair_measured, Mean_precipitation_measured_yearly, 
                              ratioSH_overall_mean, ratioSH_smallerthan_mean, ratioSH_largerthan_mean,
                              ratioSH_within1_stdev_mean, ratioSH_smallerthan1_stdev_mean, 
                              ratioSH_largerthan1_stdev_mean,ratioSH_smallerthan2_stdev_mean, 
                              ratioSH_largerthan2_stdev_mean, color1, energybalance, vegetation[0]]) 
            resultsNEE.append([mname, mean_tair_measured, Mean_precipitation_measured_yearly, 
                              ratioNEE_overall_mean, ratioNEE_smallerthan_mean, ratioNEE_largerthan_mean,
                              ratioNEE_within1_stdev_mean, ratioNEE_smallerthan1_stdev_mean, 
                              ratioNEE_largerthan1_stdev_mean, ratioNEE_smallerthan2_stdev_mean, 
                              ratioNEE_largerthan2_stdev_mean, color1, energybalance, vegetation[0]])  
    
                #Making pandas dataframes
            df_resultsLH = pd.DataFrame(resultsLH)
            df_resultsSH = pd.DataFrame(resultsSH)
            df_resultsNEE = pd.DataFrame(resultsNEE)
            
                #adding names to columns of dataframes
            df_resultsLH.columns = ['site', 'meantemp', 'meanprec',   'overall', 
                                    'smallerthanmean', 'largerthanmean', 'within1stdev', 
                                    'smallerthan1stde', 'largerthan1stdev', 
                                    'smallerthan2stdev', 'largerthan2stdev', 
                                    'color', 'energybalance', 'vegetation']
            df_resultsSH.columns = ['site', 'meantemp', 'meanprec', 'overall', 
                                    'smallerthanmean', 'largerthanmean', 'within1stdev', 
                                    'smallerthan1stdev', 'largerthan1stdev', 
                                    'smallerthan2stdev', 'largerthan2stdev', 
                                    'color', 'energybalance', 'vegetation']
            df_resultsNEE.columns = ['site', 'meantemp', 'meanprec',  'overall',
                                     'smallerthanmean', 'largerthanmean', 'within1stdev',
                                     'smallerthan1stdev', 'largerthan1stdev', 
                                     'smallerthan2stdev', 'largerthan2stdev', 'color', 
                                     'energybalance', 'vegetation']
            
            #latitudes and longitudes
            lat = ds_met['latitude'].mean()
            lon = ds_met['longitude'].mean()
            lats.append(lat)
            lons.append(lon)
            vals = df_resultsNEE.overall
            
            # creating scatterplots for temperatures below mean vs temperatures above mean. For LH, SH and NEE.
            ax = fig13.add_subplot(321)
            ax.scatter(df_resultsLH.smallerthanmean, df_resultsLH.largerthanmean, s = 20, color = df_resultsLH.color)
            
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('temperatures below mean')
            ax.set_ylabel('temperatures above mean')
            ax.set_title ('Qle above mean vs below mean')
            ax.text(-0.1,1.1, "(a)", size = 10,  horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes)
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='Annual temp above 22 degrees C')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='Annual temp below 2 degrees C')
            ax.legend(handles=[blue_patch, red_patch], loc = 'lower right')
            ax = fig13.add_subplot(323)
            ax.scatter(df_resultsSH.smallerthanmean, df_resultsSH.largerthanmean, s =20, color = df_resultsSH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('temperatures below mean')
            ax.set_ylabel('temperatures above mean')
            ax.set_title('Qh above mean vs below mean')

            ax.legend(handles=[blue_patch, red_patch], loc = 'lower right')         
            ax = fig13.add_subplot(325)
            ax.scatter(df_resultsNEE.smallerthanmean, df_resultsNEE.largerthanmean, s= 20, color = df_resultsNEE.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('temperatures below mean')
            ax.set_ylabel('temperatures above mean')
            ax.set_title('NEE above mean vs below mean') 
            ax.legend(handles=[blue_patch, red_patch], loc = 'lower right')
                # creating scatterplots for temperatures below -2stedev and above +2stdev the mean. For LH, SH and NEE.
            ax = fig13.add_subplot(322)
            ax.scatter(df_resultsLH.smallerthan2stdev, df_resultsLH.largerthan2stdev, s=20, color = df_resultsLH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('lower 2% temperatures')
            ax.set_ylabel('upper 2% temperatures')
            ax.set_title ('Qle lower tail vs upper tail')
            ax.text(-0.1,1.1, "(b)", size = 10,  horizontalalignment='center', 
                    verticalalignment='center', transform=ax.transAxes)
            ax.legend(handles=[blue_patch, red_patch], loc = 'lower right')
            ax = fig13.add_subplot(324)
            ax.scatter(df_resultsSH.smallerthan2stdev, df_resultsSH.largerthan2stdev, s = 20, color = df_resultsSH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('lower 2% temperatures')
            ax.set_ylabel('upper 2% temperatures')
            ax.set_title('Qh lower tail vs upper tail')
            ax.legend(handles=[blue_patch, red_patch], loc = 'lower right')
            ax = fig13.add_subplot(326)
            ax.scatter(df_resultsNEE.smallerthan2stdev, df_resultsNEE.largerthan2stdev, s=20, color = df_resultsNEE.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('lower 2% temperatures')
            ax.set_ylabel('upper 2% temperatures')
            ax.set_title('NEE lower tail vs upper tail') 
            ax.legend(handles=[blue_patch, red_patch], loc = 'lower right')
            

# creating histogram for the global temperature distribution 
df_temp['Total']= df_temp.iloc[:, :].sum(axis=1)
df_total = pd.DataFrame(df_temp['Total'])
df_total ["Qle"] = df_LH.iloc[:, :].sum(axis=1)
df_total ["Qh"] = df_SH.iloc[:, :].sum(axis=1)
df_total ["NEE"] = df_NEE.iloc[:, :].sum(axis=1)
#Normalizing
df_total ["Temp nor"] = df_total['Total'] / sum(df_total['Total'])
df_total ["Qle nor"] = df_total['Qle'] / sum(df_total['Total'])
df_total ["Qh nor"] = df_total['Qh'] / sum(df_total['Total'])
df_total ["NEE nor"] = df_total['NEE'] / sum(df_total['Total'])
df_total ["ratio Temp"] = 1
df_total ["ratio Qle"] = df_total ["Qle"]/ df_total["Total"]
df_total ["ratio Qh"] = df_total ["Qh"]/ df_total["Total"]
df_total ["ratio NEE"] = df_total ["NEE"]/ df_total["Total"]
x_axis= df_total.index -273

#barplot with absolute numbers
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (15,5))
ax[0].plot(x_axis, df_total['Temp nor']) 
ax[0].plot(x_axis, df_total['Qle nor'])
ax[0].plot(x_axis, df_total['Qh nor']) 
ax[0].plot(x_axis, df_total['NEE nor'])
ax[0].set_title("Number of measured temperatures, Qh, Qle, NEE (globally)") 
ax[0].set_xlabel('Temperature')
ax[0].set_ylabel('Measurements normalized') 
ax[0].legend()
ax[0].text(-0.1,1.1, "(a)", size = 12,  horizontalalignment='center', 
  verticalalignment='center', transform=ax[0].transAxes)
ax[1].plot(x_axis, df_total['ratio Temp']) 
ax[1].plot(x_axis, df_total['ratio Qle']) 
ax[1].plot(x_axis, df_total['ratio Qh'])
ax[1].plot(x_axis, df_total['ratio NEE']) 
ax[1].set_title("Ratio of measured Qle, Qh, NEE (globally)") 
ax[1].set_xlabel('Temperature')
ax[1].set_ylabel('Measurement ratio')
ax[1].legend()
ax[1].text(-0.1,1.1, "(b)", size = 12,  horizontalalignment='center', 
  verticalalignment='center', transform=ax[1].transAxes)

         
#precipitation-temperature plots
#overall performance LH, SH, NEE all vegetation
fig, ax = plt.subplots(nrows=3, ncols=3, figsize =(12,15))
plt.suptitle("Temperature-precipitation measurement ratio plots", x =0.5,y=1.02)
prectemp(df_resultsLH.meantemp-273, df_resultsLH.meanprec, 
         df_resultsLH.overall, "Qle ratio overall ",  ax[0,0], "(a)")
prectemp(df_resultsSH.meantemp-273, df_resultsSH.meanprec, df_resultsSH.overall, 
         "Qh ratio overall ",  ax[1,0], "(b) ")
prectemp(df_resultsNEE.meantemp-273, df_resultsNEE.meanprec, 
         df_resultsNEE.overall,"NEE ratio overall ",  ax[2,0], "(c)")
#above 2 stdev performance LH, SH, NEE
prectemp(df_resultsLH.meantemp-273, df_resultsLH.meanprec, 
         df_resultsLH.largerthan2stdev, "Qle ratio upper 2.2 % temperatures", ax[0,2], " ")
prectemp(df_resultsSH.meantemp-273, df_resultsSH.meanprec, 
         df_resultsSH.largerthan2stdev, "Qh ratio upper 2.2 % temperatures", ax[1,2], " ")
prectemp(df_resultsNEE.meantemp-273, df_resultsNEE.meanprec, df_resultsNEE.largerthan2stdev, 
         "NEE ratio upper 2.2 % temperatures", ax[2,2], "")
#below 2 stdev performance LH, SH, NEE
prectemp((df_resultsLH.meantemp-273), df_resultsLH.meanprec, 
         df_resultsLH.smallerthan2stdev, "Qle ratio lower 2.2 % temperatures", ax[0,1], " ")
prectemp(df_resultsSH.meantemp-273, df_resultsSH.meanprec, 
         df_resultsSH.smallerthan2stdev,"Qh ratio lower 2.2 % temperatures", ax[1,1], " ")
prectemp(df_resultsNEE.meantemp-273, df_resultsNEE.meanprec, 
         df_resultsNEE.smallerthan2stdev, "NEE ratio lower 2.2 % temperatures", ax[2,1], " ")
fig.tight_layout()
 
# creating baseplots with zoom  
#overall performance LH, SH, NEE
p1 = zoommap(df_resultsLH.overall, "Overall Qle measurement ratio", "(a)")
p2 = zoommap(df_resultsSH.overall, "Overall Qh measurement ratio", "(a)")
p3 = zoommap(df_resultsNEE.overall, "Overall NEE measurement ratio", "(a)")
#above 2 stdev performance LH, SH, NEE
p4 = zoommap(df_resultsLH.largerthan2stdev, 
             "Qle measurement ratio for the upper 2.2 % temperatures","(c)" )
p5 = zoommap(df_resultsSH.largerthan2stdev, 
             "Qh measurement ratio for the upper 2.2 % temperatures", "(c)")
p6 = zoommap(df_resultsNEE.largerthan2stdev, 
             "NEE measurement ratio for the upper 2.2 % temperatures", "(c)")
#below 2 stdev performance LH, SH, NEE
p7 = zoommap(df_resultsLH.smallerthan2stdev, 
             "Qle measurement ratio for the lower 2.2 % temperatures", "(b)")
p8 = zoommap(df_resultsSH.smallerthan2stdev, 
             "Qh measurement ratio for the lower 2.2 % temperatures", "(b)")
p9 = zoommap(df_resultsNEE.smallerthan2stdev, 
             "NEE measurement ratio for the lower 2.2 % temperatures", "(b)")




#Creating heatmaps of the results. Here, we use the same colorscheme as the graphs. These heatmaps show 
#per location the ratio of LH, SH and NEE measurements. 

sites = np.array(df_resultsLH['site'])
label = ["Overall", "Lower extreme", "Higher extreme"]
data_LH = np.array(df_resultsLH[['overall', 'smallerthan2stdev', 'largerthan2stdev']])
data_SH = np.array(df_resultsSH[['overall', 'smallerthan2stdev', 'largerthan2stdev']])
data_NEE = np.array(df_resultsNEE[['overall', 'smallerthan2stdev', 'largerthan2stdev']])
cmap = matplotlib.colors.ListedColormap(['#d73027', '#fc8d59', '#fee090', 
                                             '#e0f3f8', '#91bfdb', '#4575b4'])
norm = matplotlib.colors.BoundaryNorm([0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0], cmap.N)

fig, ax = plt.subplots(nrows = 1, ncols = 7, figsize = (12,8))
plt.suptitle("Measurement ratios of Qle", x =0.5, y = 0.94)
im =heatmap(ax[0], data_LH[0:31][:], sites[0:31][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[1], data_LH[31:62][:], sites[31:62][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[2], data_LH[62:93][:], sites[62:93][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[3], data_LH[93:124][:], sites[93:124][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[4], data_LH[124:155][:], sites[124:155][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[5], data_LH[155:186][:], sites[155:186][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[6], data_LH[186:217][:], sites[186:217][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
fig.subplots_adjust( bottom=0.1, right=0.9, top=0.9, wspace=0.7, hspace=0)
cax = plt.axes([0.92, 0.1, 0.01, 0.8]) 
plt.colorbar(im, cax=cax, spacing = 'proportional')
plt.show()

fig, ax = plt.subplots(nrows = 1, ncols = 7, figsize = (12,8))
plt.suptitle("Measurement ratios of Qh", x =0.5, y = 0.94)
im = heatmap(ax[0], data_SH[0:31][:], sites[0:31][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[1], data_SH[31:62][:], sites[31:62][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[2], data_SH[62:93][:], sites[62:93][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[3], data_SH[93:124][:], sites[93:124][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[4], data_SH[124:155][:], sites[124:155][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[5], data_SH[155:186][:], sites[155:186][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[6], data_SH[186:217][:], sites[186:217][:], label, cmap, norm) 
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
fig.subplots_adjust( bottom=0.1, right=0.9, top=0.9, wspace=0.7, hspace=0)
cax = plt.axes([0.92, 0.1, 0.01, 0.8]) 
plt.colorbar(im, cax=cax, spacing = 'proportional')  
plt.show()

fig, ax = plt.subplots(nrows = 1, ncols = 7, figsize = (12,8))
plt.suptitle("Measurement ratios of NEE", x =0.5, y = 0.94)
im = heatmap(ax[0], data_NEE[0:31][:], sites[0:31][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[1], data_NEE[31:62][:], sites[31:62][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[2], data_NEE[62:93][:], sites[62:93][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[3], data_NEE[93:124][:], sites[93:124][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[4], data_NEE[124:155][:], sites[124:155][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[5], data_NEE[155:186][:], sites[155:186][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
im = heatmap(ax[6], data_NEE[186:217][:], sites[186:217][:], label, cmap, norm)
annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func))
fig.subplots_adjust( bottom=0.1, right=0.9, top=0.9, wspace=0.7, hspace=0)
cax = plt.axes([0.92, 0.1, 0.01, 0.8]) 
plt.colorbar(im, cax=cax, spacing = 'proportional')
plt.show()

#Selecting the most suitable sites
lons = pd.DataFrame(lons, columns =['lon'])
lats = pd.DataFrame(lats, columns = ['lat'])


#Renaming and putting DataFrames together
df_resultsLH = df_resultsLH.rename (columns={'overall': 'overallLH', 
                                 'smallerthan2stdev': 'smallerthan2stdevLH', 
                                 'largerthan2stdev': 'largerthan2stdevLH'})
df_resultsSH = df_resultsSH.rename (columns={'overall': 'overallSH', 
                                 'smallerthan2stdev': 'smallerthan2stdevSH', 
                                 'largerthan2stdev': 'largerthan2stdevSH'})
df_resultsNEE = df_resultsNEE.rename (columns={'overall': 'overallNEE', 
                                 'smallerthan2stdev': 'smallerthan2stdevNEE', 
                                 'largerthan2stdev': 'largerthan2stdevNEE'})    
df_results = result = pd.concat([df_resultsLH.site, df_resultsLH.overallLH, df_resultsLH.smallerthan2stdevLH,
                                 df_resultsLH.largerthan2stdevLH,df_resultsSH.overallSH, 
                                 df_resultsSH.smallerthan2stdevSH,df_resultsSH.largerthan2stdevSH,
                                 df_resultsNEE.overallNEE, df_resultsNEE.smallerthan2stdevNEE,
                                 df_resultsNEE.largerthan2stdevNEE, lons, lats], axis = 1)

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
overall2ndbestsitesall['color'] = '#e0f3f8'
#SH and LH above 90%
overallbestsitesSHLH = pd.concat([overallbestsitesSHLH.site, overallbestsitesSHLH.overallLH, 
                      overallbestsitesSHLH.overallSH, overallbestsitesSHLH.overallNEE,
                      overallbestsitesSHLH.lon, overallbestsitesSHLH.lat], axis =1)
overallbestsitesSHLH['color'] = 'orange'
#SH and LH above 80%
overall2ndbestsitesSHLH = pd.concat([overall2ndbestsitesSHLH.site, overall2ndbestsitesSHLH.overallLH, 
                      overall2ndbestsitesSHLH.overallSH, overall2ndbestsitesSHLH.overallNEE,
                      overall2ndbestsitesSHLH.lon, overall2ndbestsitesSHLH.lat], axis =1)
overall2ndbestsitesSHLH['color'] = 'yellow'

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
lightblue_patch = mpatches.Patch(color='#e0f3f8', label='LH, SH and NEE above 0.8')
orange_patch = mpatches.Patch(color='orange', label='LH, SH above 0.9')
yellow_patch = mpatches.Patch(color='yellow', label='LH, SH above 0.8')
ax.legend(handles=[blue_patch, lightblue_patch, orange_patch, yellow_patch], bbox_to_anchor = (1.25, 1.9, 0.3, 0.3))


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
lower2ndbestsitesall['color'] = '#e0f3f8'
#SH and LH above 90%
lowerbestsitesSHLH = pd.concat([lowerbestsitesSHLH.site, lowerbestsitesSHLH.smallerthan2stdevLH, 
                      lowerbestsitesSHLH.smallerthan2stdevSH, lowerbestsitesSHLH.smallerthan2stdevNEE,
                      lowerbestsitesSHLH.lon, lowerbestsitesSHLH.lat], axis =1)
lowerbestsitesSHLH['color'] = 'orange'
#SH and LH above 80%
lower2ndbestsitesSHLH = pd.concat([lower2ndbestsitesSHLH.site, lower2ndbestsitesSHLH.smallerthan2stdevLH, 
                      lower2ndbestsitesSHLH.smallerthan2stdevSH, lower2ndbestsitesSHLH.smallerthan2stdevNEE,
                      lower2ndbestsitesSHLH.lon, lower2ndbestsitesSHLH.lat], axis =1)
lower2ndbestsitesSHLH['color'] = 'yellow'
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
ax.legend(handles=[blue_patch, lightblue_patch, orange_patch, yellow_patch],
          bbox_to_anchor = (1.25, 1.2, 0.3, 0.3))

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
upper2ndbestsitesall['color'] = '#e0f3f8'
#SH and LH above 90%
upperbestsitesSHLH = pd.concat([upperbestsitesSHLH.site, upperbestsitesSHLH.largerthan2stdevLH, 
                      upperbestsitesSHLH.largerthan2stdevSH, upperbestsitesSHLH.largerthan2stdevNEE,
                      upperbestsitesSHLH.lon, upperbestsitesSHLH.lat], axis =1)
upperbestsitesSHLH['color'] = 'orange'
#SH and LH above 80%
upper2ndbestsitesSHLH = pd.concat([upper2ndbestsitesSHLH.site, upper2ndbestsitesSHLH.largerthan2stdevLH, 
                      upper2ndbestsitesSHLH.largerthan2stdevSH, upper2ndbestsitesSHLH.largerthan2stdevNEE,
                      upper2ndbestsitesSHLH.lon, upper2ndbestsitesSHLH.lat], axis =1)
upper2ndbestsitesSHLH['color'] = 'yellow'
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
ax.legend(handles=[blue_patch, lightblue_patch, orange_patch, yellow_patch], 
          bbox_to_anchor = (1.25, 2.6, 0.3, 0.3))

#Show the results of the table in basemaps
#Baseplot for overall performance
fig =  plt.figure(figsize=(12,8))
plt.title("Overall suitable sites")   
plt.text(-0.01,1.3, "(a)", size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)
plt.legend(handles=[blue_patch, lightblue_patch, orange_patch, yellow_patch], loc = 3) 
m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
            urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
m.drawcoastlines()
for row in table_overall.itertuples():
    x,y = m(table_overall.lon.values, table_overall.lat.values)
    m.scatter(x,y, s = 30, color = table_overall.color.values)
    
#Baseplot for lower extreme performance
fig =  plt.figure(figsize=(12,8))
plt.title("Lower extreme suitable sites")  
plt.text(-0.01,1.3, "(b)", size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)
plt.legend(handles=[blue_patch, lightblue_patch, orange_patch, yellow_patch], loc = 3)  
m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
            urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
m.drawcoastlines()
for row in table_lower.itertuples():
    x,y = m(table_lower.lon.values, table_lower.lat.values)
    m.scatter(x,y, s = 30, color = table_lower.color.values)
    
#Baseplot for upper extreme performance
fig =  plt.figure(figsize=(12,8))
plt.title("Upper extreme suitable sites") 
plt.text(-0.01,1.3, "(c)", size = 12,  horizontalalignment='center', 
     verticalalignment='center', transform=ax.transAxes)
#plt.legend(handles=[blue_patch, lightblue_patch, orange_patch, yellow_patch], loc = 3)   
m = Basemap(projection = 'mill', llcrnrlat = -45, llcrnrlon = -160, 
            urcrnrlat= 82, urcrnrlon = 170, resolution = 'l')
m.drawcoastlines()

for row in table_upper.itertuples():
    x,y = m(table_upper.lon.values, table_upper.lat.values)
    m.scatter(x,y, s = 30, color = table_upper.color.values)





#Figures that I am not currently using:
"""
# scatterplot on quality control vs. energybalance closure overall performance
fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (12,16))
ax[0,0].scatter(df_resultsLH.overall, df_resultsLH.energybalance)
ax[0,0].set_title("LH overall")
ax[0,0].set_xlabel("performance LH measurement")
ax[0,0].set_ylabel("residual energy (W/m2)")
ax[1,0].scatter (df_resultsSH.overall, df_resultsSH.energybalance)
ax[1,0].set_title("SH overall")
ax[1,0].set_xlabel("performance LH measurement")
ax[1,0].set_ylabel("residual energy (W/m2)")
ax[2,0].scatter (df_resultsNEE.overall, df_resultsNEE.energybalance)
ax[2,0].set_title("NEE overall")
ax[2,0].set_xlabel("performance LH measurement")
ax[2,0].set_ylabel("residual energy (W/m2)")


#scatterplot on quality control vs. energybalance closure above 2 stdev performance
ax[0,1].scatter(df_resultsLH.largerthan2stdev, df_resultsLH.energybalance)
ax[0,1].set_title("LH above 2 stdev from the mean")
ax[0,1].set_xlabel("performance LH measurement")
ax[0,1].set_ylabel("residual energy (W/m2)")
ax[1,1].scatter (df_resultsSH.largerthan2stdev, df_resultsSH.energybalance)
ax[1,1].set_title("SH above 2 stdev from the mean")
ax[1,1].set_xlabel("performance LH measurement")
ax[1,1].set_ylabel("residual energy (W/m2)")
ax[2,1].scatter (df_resultsNEE.largerthan2stdev, df_resultsNEE.energybalance)
ax[2,1].set_title("NEE above 2 stdev from the mean")
ax[2,1].set_xlabel("performance LH measurement")
ax[2,1].set_ylabel("residual energy (W/m2)")


#scatterplot on quality control vs. energybalance closure below 2 stdev performance
ax[0,2].scatter(df_resultsLH.smallerthan2stdev, df_resultsLH.energybalance)
ax[0,2].set_title("LH below 2 stdev from the mean")
ax[0,2].set_xlabel("performance LH measurement")
ax[0,2].set_ylabel("residual energy (W/m2)")
ax[1,2].scatter (df_resultsSH.smallerthan2stdev, df_resultsSH.energybalance)
ax[1,2].set_title("SH below 2 stdev from the mean")
ax[1,2].set_xlabel("performance LH measurement")
ax[1,2].set_ylabel("residual energy (W/m2)")
ax[2,2].scatter (df_resultsNEE.smallerthan2stdev, df_resultsNEE.energybalance)
ax[2,2].set_title("NEE below 2 stdev from the mean")
ax[2,2].set_xlabel("performance LH measurement")
ax[2,2].set_ylabel("residual energy (W/m2)")
plt.show()
#precipitation- temperature plots for vegetation
#Forests
fig, ax = plt.subplots(nrows=3, ncols=3, figsize =(12,15)) 
plt.suptitle("Temperature-precipitation measurement ratio plots forests", x =0.5,y=1.02)
forests_LH = df_resultsLH[df_resultsLH['vegetation'].str.contains("Forests")]
forests_SH = df_resultsSH[df_resultsSH['vegetation'].str.contains("Forests")]
forests_NEE = df_resultsNEE[df_resultsNEE['vegetation'].str.contains("Forests")]
prectemp(forests_LH.meantemp-273, forests_LH.meanprec, forests_LH.overall, 
         "LH ratio overall ",  ax[0,0])
prectemp(forests_LH.meantemp-273, forests_LH.meanprec, forests_LH.smallerthan2stdev, 
         "LH ratio lower 2.2 % temperatures",  ax[0,1])
prectemp(forests_LH.meantemp-273, forests_LH.meanprec, forests_LH.largerthan2stdev, 
         "LH ratio upper 2.2 % temperatures",  ax[0,2])
prectemp(forests_SH.meantemp-273, forests_SH.meanprec, forests_SH.overall, 
         "SH ratio overall ",  ax[1,0])
prectemp(forests_SH.meantemp-273, forests_SH.meanprec, forests_SH.smallerthan2stdev, 
         "SH ratio lower 2.2 % temperatures",  ax[1,1])
prectemp(forests_SH.meantemp-273, forests_SH.meanprec, forests_SH.largerthan2stdev, 
         "SH ratio upper 2.2 % temperatures",  ax[1,2])
prectemp(forests_NEE.meantemp-273, forests_NEE.meanprec, forests_NEE.overall, 
         "NEE ratio overall ",  ax[2,0])
prectemp(forests_NEE.meantemp-273, forests_NEE.meanprec, forests_NEE.smallerthan2stdev, 
         "NEE ratio lower 2.2 % temperatures",  ax[2,1])
prectemp(forests_NEE.meantemp-273, forests_NEE.meanprec, forests_NEE.largerthan2stdev, 
         "NEE ratio upper 2.2 % temperatures",  ax[2,2])
fig.tight_layout()
#grasslands & croplands
#LH grasslands
fig, ax = plt.subplots(nrows=3, ncols=3, figsize =(12,15)) 
plt.suptitle("Temperature-precipitation measurement ratio plots grasslands", x =0.5,y=1.02)
grasslands_LH = df_resultsLH[df_resultsLH['vegetation'].str.contains('Grass|Crop')]
grasslands_SH = df_resultsSH[df_resultsSH['vegetation'].str.contains('Grass|Crop')]
grasslands_NEE = df_resultsNEE[df_resultsLH['vegetation'].str.contains('Grass|Crop')]
prectemp(grasslands_LH.meantemp-273, grasslands_LH.meanprec, grasslands_LH.overall, 
         "LH ratio overall ",  ax[0,0])
prectemp(grasslands_LH.meantemp-273, grasslands_LH.meanprec, grasslands_LH.smallerthan2stdev, 
         "LH ratio lower 2.2 % temperatures",  ax[0,1])
prectemp(grasslands_LH.meantemp-273, grasslands_LH.meanprec, grasslands_LH.largerthan2stdev, 
         "LH ratio upper 2.2 % temperatures",  ax[0,2])
prectemp(grasslands_SH.meantemp-273, grasslands_SH.meanprec, grasslands_SH.overall, 
         "SH ratio overall ",  ax[1,0])
prectemp(grasslands_SH.meantemp-273, grasslands_SH.meanprec, grasslands_SH.smallerthan2stdev, 
         "SH ratio lower 2.2 % temperatures",  ax[1,1])
prectemp(grasslands_SH.meantemp-273, grasslands_SH.meanprec, grasslands_SH.largerthan2stdev, 
         "SH ratio upper 2.2 % temperatures",  ax[1,2])
prectemp(grasslands_NEE.meantemp-273, grasslands_NEE.meanprec, grasslands_NEE.overall, 
         "NEE ratio overall ",  ax[2,0])
prectemp(grasslands_NEE.meantemp-273, grasslands_NEE.meanprec, grasslands_NEE.smallerthan2stdev, 
         "NEE ratio lower 2.2 % temperatures",  ax[2,1])
prectemp(grasslands_NEE.meantemp-273, grasslands_NEE.meanprec, grasslands_NEE.largerthan2stdev, 
         "NEE ratio upper 2.2 % temperatures",  ax[2,2])
fig.tight_layout()
"""


"""
            #barplot temp distribution with different temperatures
            list_of_ranges = [
                            ('fig1b', 270, 275, 421),
                            ('fig1c', 275, 280, 422),
                            ('fig1d', 280, 285, 423),
                            ('fig1e', 285, 290, 424),
                            ('fig1f', 290, 295, 425),
                            ('fig1g', 295, 300, 426),
                            ('fig1h', 300, 9999999, 427)]
            
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= ds_met.Tair.mean() < maximum:
                    ax = fig15.add_subplot(otherid)
                    dftemp_sorted.plot(y= [0], ax =ax, legend = False)
                    title_label = "temperature distribution"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("Temperature")
                    ax.set_ylabel("Frequency")
            fig15.tight_layout()
            #fig1.savefig("LH ratio and temperature stdev.pdf")
"""      
"""
            #plot with Ratio of latent heat measurements plot of all sites, with a stdev
            #scale on the x-axis
            ax = fig10.add_subplot(111)
            ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'ratioLH'])
            ax.set_title("Latent heat measurements ratio")
            ax.set_xlabel("Temperature stdev")
            ax.set_ylabel("Ratio")
            
            #plot with ratio of sensible heat measurements plot of all sites, with a stdev 
            #scale on the x-axis
            ax = fig11.add_subplot(111)
            ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'ratioSH'])
            ax.set_title("Sensible heat measurements ratio")
            ax.set_xlabel("Temperature stdev")
            ax.set_ylabel("Ratio")
            
            #plot with ratio of NEE measurements plot of all sites, with a stdev scale on 
            #the x-axis
            ax = fig12.add_subplot(111)
            ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'ratioNEE'])
            ax.set_title("NEE measurements ratio")
            ax.set_xlabel("Temperature stdev")
            ax.set_ylabel("Ratio")
            
"""           
     
""" 
                # creating scatterplots for temperatures below mean vs temperatures above mean. For LH, SH and NEE.
            ax = fig13.add_subplot(321)
            ax.scatter(df_resultsLH.smallerthanmean, df_resultsLH.largerthanmean, s = 20, color = df_resultsLH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('temperatures below mean')
            ax.set_ylabel('temperatures above mean')
            ax.set_title ('LH above mean vs below mean')
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='mean annual temperature > 22')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='mean temperature < 2')
            ax.legend(handles=[blue_patch, red_patch])
            ax = fig13.add_subplot(323)
            ax.scatter(df_resultsSH.smallerthanmean, df_resultsSH.largerthanmean, s =20, color = df_resultsSH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('temperatures below mean')
            ax.set_ylabel('temperatures above mean')
            ax.set_title('SH above mean vs below mean')
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='mean annual temperature > 22')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='mean temperature < 2')
            ax.legend(handles=[blue_patch, red_patch])         
            ax = fig13.add_subplot(325)
            ax.scatter(df_resultsNEE.smallerthanmean, df_resultsNEE.largerthanmean, s= 20, color = df_resultsNEE.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('temperatures below mean')
            ax.set_ylabel('temperatures above mean')
            ax.set_title('NEE above mean vs below mean') 
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='mean annual temperature > 22')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='mean temperature < 2')
            ax.legend(handles=[blue_patch, red_patch])
                # creating scatterplots for temperatures below -2stedev and above +2stdev the mean. For LH, SH and NEE.
            ax = fig13.add_subplot(322)
            ax.scatter(df_resultsLH.smallerthan2stdev, df_resultsLH.largerthan2stdev, s=20, color = df_resultsLH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('lower 2% temperatures')
            ax.set_ylabel('upper 2% temperatures')
            ax.set_title ('LH lower tail vs upper tail')
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='mean annual temperature > 22')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='mean temperature < 2')
            ax.legend(handles=[blue_patch, red_patch])
            ax = fig13.add_subplot(324)
            ax.scatter(df_resultsSH.smallerthan2stdev, df_resultsSH.largerthan2stdev, s = 20, color = df_resultsSH.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('lower 2% temperatures')
            ax.set_ylabel('upper 2% temperatures')
            ax.set_title('SH lower tail vs upper tail')
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='mean annual temperature > 22')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='mean temperature < 2')
            ax.legend(handles=[blue_patch, red_patch])
            ax = fig13.add_subplot(326)
            ax.scatter(df_resultsNEE.smallerthan2stdev, df_resultsNEE.largerthan2stdev, s=20, color = df_resultsNEE.color)
            ax.plot( [0,1],[0,1] )
            ax.set_xlabel('lower 2% temperatures')
            ax.set_ylabel('upper 2% temperatures')
            ax.set_title('NEE lower tail vs upper tail') 
            red_patch = mpatches.Circle((0, 0), color='Firebrick', label='mean annual temperature > 22')
            blue_patch = mpatches.Circle((0, 0), radius = 0.25, color='blue', label='mean temperature < 2')
            ax.legend(handles=[blue_patch, red_patch])
"""

 
#USE THIS PIECE OF CODE TO MAKE RATIO OF LH, SH and NEE measurement PLOTS WITH STDEV ON THE X-AXIS 
#DIFFERENTIATE BETWEEN MEAN TEMPERATURE, PRECIPITATION AND VEGETATION.
"""
                #Latent heat measurement plots with different site mean temperatures  categories
            list_of_ranges = [
                            ('fig1b', 270, 275, 421),
                            ('fig1c', 275, 280, 422),
                            ('fig1d', 280, 285, 423),
                            ('fig1e', 285, 290, 424),
                            ('fig1f', 290, 295, 425),
                            ('fig1g', 295, 300, 426),
                            ('fig1h', 300, 9999999, 427)]       
                       
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= ds_met.Tair.mean() < maximum:
                    ax = fig1.add_subplot(otherid)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageLH'])
                    title_label = "ratio LH Tair"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio")
            fig1.tight_layout()
            fig1.savefig("LH ratio and temperature stdev.pdf")
            
            #Latent heat measurement plots with different site mean precipitation categories
            list_of_ranges = [
                            ('fig4a', 0, 0.2, 421),
                            ('fig4b', 0.2, 0.4, 422),
                            ('fig4c', 0.4, 0.6, 423),
                            ('fig4d', 0.6, 0.8, 424),
                            ('fig4e', 0.8, 1.0, 425),
                            ('fig4f', 1.0, 1.2, 426),
                            ('fig4g', 1.2, 1.4, 427),
                            ('fig4h', 1.4, 9999999, 428)] 
                    
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= mean_precipitation_yearly < maximum:
                    ax = fig2.add_subplot(otherid)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageLH'])
                    title_label = "ratio LH precipitation"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio")   
            fig2.tight_layout()
            fig2.savefig("LH ratio and precipitation stdev.pdf")
            
                #Latent heat measurement plots with different site vegetation categories
            list_of_ranges = [
                            ('fig7a', "Savannas", 7,2,1),
                            ('fig7b', "Evergreen Needleleaf Forests",7,2,2),
                            ('fig7c', "Cropland/Natural Vegetation Mosaic",7,2,3),
                            ('fig7d', "Evergreen Broadleaf Forests",7,2,4),
                            ('fig7e', "Deciduous Broadleaf Forests",7,2,5),
                            ('fig7f', "Grasslands",7,2,6),
                            ('fig7g', "Open Shrublands",7,2,7),
                            ('fig7h', "Permanent Wetlands",7,2,8),
                            ('fig7i', "Croplands",7,2,9),
                            ('fig7j', "Mixed Forests",7,2,10),
                            ('fig7k', "Woody Savannas",7,2,11),
                            ('fig7l', "Closed Shrublands",7,2,12),
                            ('fig7m', "Snow and Ice",7,2,13)]
                            
            decoded = ds_met['IGBP_veg_long'].str.decode("utf-8")
                        
            for (name, vegetation, left, right,number) in list_of_ranges:
                if str(decoded[0]).strip() == vegetation:
                    ax = fig3.add_subplot(left, right, number)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageLH'])
                    title_label = "ratio LH vegetation" + " " + str(vegetation)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio") 
            fig3.tight_layout()    
            fig3.savefig("LH ratio and vegetation stdev.pdf")    
                                          
                # Sensible heat measurement plots with different site mean temperature categories
            list_of_ranges = [
                            
                            ('fig1b', 270, 275, 421),
                            ('fig1c', 275, 280, 422),
                            ('fig1d', 280, 285, 423),
                            ('fig1e', 285, 290, 424),
                            ('fig1f', 290, 295, 425),
                            ('fig1g', 295, 300, 426),
                            ('fig1h', 300, 9999999, 427)]       
            
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= ds_met.Tair.mean() < maximum:
                    ax = fig4.add_subplot(otherid)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageSH'])
                    title_label = "ratio SH Tair"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio")
            fig4.tight_layout()
            fig4.savefig("SH ratio and temperature stdev.pdf") 
            
                # Sensible heat measurement plots with different site mean precipitation categories
            list_of_ranges = [
                            ('fig4a', 0, 0.2, 421),
                            ('fig4b', 0.2, 0.4, 422),
                            ('fig4c', 0.4, 0.6, 423),
                            ('fig4d', 0.6, 0.8, 424),
                            ('fig4e', 0.8, 1.0, 425),
                            ('fig4f', 1.0, 1.2, 426),
                            ('fig4g', 1.2, 1.4, 427),
                            ('fig4h', 1.4, 9999999, 428)] 
                    
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= mean_precipitation_yearly < maximum:
                    ax = fig5.add_subplot(otherid)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageSH'])
                    title_label = "ratio SH precipitation"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio") 
            fig5.tight_layout()        
            fig5.savefig("SH ratio and precipitation stdev.pdf")

                # Sensible heat measurement plots with different site vegetation
            list_of_ranges = [
                            ('fig7a', "Savannas", 7,2,1),
                            ('fig7b', "Evergreen Needleleaf Forests",7,2,2),
                            ('fig7c', "Cropland/Natural Vegetation Mosaic",7,2,3),
                            ('fig7d', "Evergreen Broadleaf Forests",7,2,4),
                            ('fig7e', "Deciduous Broadleaf Forests",7,2,5),
                            ('fig7f', "Grasslands",7,2,6),
                            ('fig7g', "Open Shrublands",7,2,7),
                            ('fig7h', "Permanent Wetlands",7,2,8),
                            ('fig7i', "Croplands",7,2,9),
                            ('fig7j', "Mixed Forests",7,2,10),
                            ('fig7k', "Woody Savannas",7,2,11),
                            ('fig7l', "Closed Shrublands",7,2,12),
                            ('fig7m', "Snow and Ice",7,2,13)]
            
            for (name, vegetation, left, right, number) in list_of_ranges:
                if str(decoded[0]).strip() == vegetation:
                    ax = fig6.add_subplot(left,right,number)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageSH'])
                    title_label = "ratio SH vegetation" + " " + str(vegetation)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio") 
            fig6.tight_layout()
            fig6.savefig("SH ratio and vegetation stdev.pdf")
                                  
                # NEE measurement plots with different site mean temperature categories
            list_of_ranges = [                            
                            ('fig1b', 270, 275, 421),
                            ('fig1c', 275, 280, 422),
                            ('fig1d', 280, 285, 423),
                            ('fig1e', 285, 290, 424),
                            ('fig1f', 290, 295, 425),
                            ('fig1g', 295, 300, 426),
                            ('fig1h', 300, 9999999, 427)]      
            
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= ds_met.Tair.mean() < maximum:
                    ax = fig7.add_subplot(otherid)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageNEE'])
                    title_label = "ratio NEE Tair"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio")   
            fig7.tight_layout()
            fig7.savefig("NEE ratio and temperature stdev.pdf")
            
                # NEE measurement plots with different site mean precipitation categories
            list_of_ranges = [
                            ('fig4a', 0, 0.2, 421),
                            ('fig4b', 0.2, 0.4, 422),
                            ('fig4c', 0.4, 0.6, 423),
                            ('fig4d', 0.6, 0.8, 424),
                            ('fig4e', 0.8, 1.0, 425),
                            ('fig4f', 1.0, 1.2, 426),
                            ('fig4g', 1.2, 1.4, 427),
                            ('fig4h', 1.4, 9999999, 428)] 
                    
            for (name, minimum, maximum, otherid) in list_of_ranges:
                if minimum <= mean_precipitation_yearly < maximum:
                    ax = fig8.add_subplot(otherid)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageNEE'])
                    title_label = "ratio NEE precipitation"
                    if minimum == 0:
                        title_label = title_label + " <" + str(maximum)
                    elif maximum == 9999999:
                        title_label = title_label + " >" + str(minimum)
                    else:
                        title_label = title_label + " " + str(minimum) + "-" + str(maximum)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio")
            fig8.tight_layout()
            fig8.savefig("NEE ratio and precipitation stdev.pdf")        
                    
                    # NEE measurement plots with different site vegetation
            list_of_ranges = [
                            ('fig7a', "Savannas", 7,2,1),
                            ('fig7b', "Evergreen Needleleaf Forests",7,2,2),
                            ('fig7c', "Cropland/Natural Vegetation Mosaic",7,2,3),
                            ('fig7d', "Evergreen Broadleaf Forests",7,2,4),
                            ('fig7e', "Deciduous Broadleaf Forests",7,2,5),
                            ('fig7f', "Grasslands",7,2,6),
                            ('fig7g', "Open Shrublands",7,2,7),
                            ('fig7h', "Permanent Wetlands",7,2,8),
                            ('fig7i', "Croplands",7,2,9),
                            ('fig7j', "Mixed Forests",7,2,10),
                            ('fig7k', "Woody Savannas",7,2,11),
                            ('fig7l', "Closed Shrublands",7,2,12),
                            ('fig7m', "Snow and Ice",7,2,13)]
            
            for (name, vegetation, left, right, number) in list_of_ranges:
                if str(decoded[0]).strip() == vegetation:
                    ax = fig9.add_subplot(left,right,number)
                    ax.plot(dftemp_sorted.loc[:,"stdevscale"],dftemp_sorted.loc[:,'percentageNEE'])
                    title_label = "ratio NEE vegetation" + " " + str(vegetation)
                    ax.set_title(title_label)
                    ax.set_xlabel("temperature stdev")
                    ax.set_ylabel("ratio") 
            fig9.tight_layout()
            fig9.savefig("NEE ratio and vegetation stdev.pdf")
    
    #to show which sites are not suitable
    else: 
        print("not suitable")
"""            
      
            
            
            