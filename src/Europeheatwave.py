#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 21 15:02:14 2018

@author: z5228341
"""
    # Import packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import pandas as pd
import glob
import xarray as xr
import matplotlib.backends.backend_pdf #import PdfPages
import os
import matplotlib
        

#Import the files
files_met = sorted(glob.glob("/home/z5228341/Desktop/FLUXNET & LaThuille/MET/Met/*"))

files_flux = sorted(glob.glob("/home/z5228341/Desktop/FLUXNET & LaThuille/FLUX/Flux/*"))


#function to open and organize data
def open_file(fname):
    ds = xr.open_dataset(fname)
    ds = ds.squeeze(dim=["x","y"], drop=True).to_dataframe()
    ds = ds.reset_index()
    ds = ds.set_index('time') 
    return (ds)
 
def Europe(vals, title, ax):

    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,
                        wspace=0.15,hspace=0.05)
    m = Basemap(ax = ax, projection = 'mill', llcrnrlat = 35, llcrnrlon = -10, 
                urcrnrlat= 60, urcrnrlon = 20, resolution = 'l')
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth = 0.5)
    plt.subplots_adjust(left=0.05,right=0.95,top=0.90,bottom=0.05,
                        wspace=0.15,hspace=0.05)
    cmap = matplotlib.colors.ListedColormap(['#d73027', '#fc8d59', '#fee090', 
                                             '#e0f3f8', '#91bfdb', '#4575b4'])
    bounds = [0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]
    norm = matplotlib.colors.BoundaryNorm(bounds, cmap.N)
    sc = m.scatter(x, y, s = 40, c = vals, cmap=cmap, norm=norm)
    plt.colorbar(sc, ax = ax, orientation="vertical", boundaries = bounds, 
                 spacing = 'proportional', 
                 ticks = bounds)
    ax.set_title(title)    
    return (fig)
 

# empty lists to add data in for the table/figures
results_LH_hw = []
results_SH_hw = []
results_NEE_hw = []
results_LH_nohw = []
results_SH_nohw = []
results_NEE_nohw = []
lons=[]
lats=[]


#creation of for loop which analyzes each Fluxdataset
for m,f in zip(files_met, files_flux):
    
      
    #Opening the files 
    ds_met = open_file(m)    
    ds_flux = open_file(f)
    
    #only selecting the months July and August
    ds_met = ds_met.loc[(ds_met.index.month==7) | (ds_met.index.month==8)]
    ds_flux = ds_flux.loc[(ds_flux.index.month==7) | (ds_flux.index.month==8)]
    
    
    Tair_measured = ds_met[ds_met.Tair_qc < 1] 
    
    lat = ds_met['latitude'].mean()
    lon = ds_met['longitude'].mean()
    if (ds_met.index == pd.Timestamp("2003-07-01 00:00:00")).any() and\
    len(ds_met) > 1000 and (len(Tair_measured)/len(ds_met))>0.5 \
    and lat > 35 and lat <60 and lon >-10 and lon <20:
        print(m,f)
        
        
        ds_met_hw = ds_met['2003/07/01' : '2003/08/31'] #ONLY WITH EURO HEATWAVE
        ds_flux_hw = ds_flux['2003/07/01' : '2003/08/31'] #ONLY WITH EURO HEATWAVE
        ds_met_nohw_before2003 =  ds_met[:'2003/06/30']
        ds_met_nohw_after2003 = ds_met['2003/09/01':]
        ds_met_nohw = pd.concat([ds_met_nohw_before2003, ds_met_nohw_after2003])
        ds_flux_nohw_before2003 =  ds_flux[:'2003/06/30']
        ds_flux_nohw_after2003 = ds_flux['2003/09/01':]
        ds_flux_nohw = pd.concat([ds_flux_nohw_before2003, ds_flux_nohw_after2003])
        
        
        #latitudes and longitudes
        
        lats.append(lat)
        lons.append(lon)
        
        ds_met = ds_met_hw
        ds_flux = ds_flux_hw           
       
        
        Mean_precipitation_measured_yearly = ((ds_met.Rainf.mean())*48*365) #not all datasets had a qc for this. 
        mean_tair_measured = np.mean(Tair_measured.Tair)
        stdev_tair_measured = np.std(Tair_measured.Tair)
        # Filter to measured temperature (qc=0) and only shortwave 
        #incoming radiation >0 and filtering out data between 11 pm and 6 am
                        
        ds_flux = ds_flux[ds_met.Tair_qc < 1]
        ds_met = ds_met[ds_met.Tair_qc < 1]
        ds_flux = ds_flux[ds_met.SWdown > 1]
        ds_met = ds_met[ds_met.SWdown > 1]
        ds_flux = ds_flux.drop(ds_flux.between_time("23:00", "6:00").index)
        ds_met = ds_met.drop(ds_met.between_time("23:00", "6:00").index)
 
        if len(ds_met.Tair >1):            
            #creating bins of 1 K and assigning a temperature to a bin. 
            
            minimum_tair = min(ds_met.Tair)
            maximum_tair = max(ds_met.Tair)
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
            
            #shorten the filename to first 6 letters, which is the abberviation 
            #for the site
            mname = os.path.splitext(os.path.basename(m))[0]
            mname=mname[0:6]
            
        else:
            ratioLH_overall_mean = np.nan
            ratioSH_overall_mean = np.nan
            ratioNEE_overall_mean = np.nan
            
                #shorten the filename to first 6 letters, which is the abberviation 
            #for the site
            mname = os.path.splitext(os.path.basename(m))[0]
            mname=mname[0:6]
               
        #Adding the results to the empty lists
        results_LH_hw.append([mname, mean_tair_measured,
                          ratioLH_overall_mean])                          
        results_SH_hw.append([mname, mean_tair_measured, 
                          ratioSH_overall_mean]) 
        results_NEE_hw.append([mname, mean_tair_measured, 
                          ratioNEE_overall_mean])

       
        ds_met = ds_met_nohw
        ds_flux = ds_flux_nohw
 
        mean_tair_measured = np.mean(Tair_measured.Tair)
        stdev_tair_measured = np.std(Tair_measured.Tair)
        
        #Does the data have enough recorded temperatures? First, selecting 
        #only measurements with qc = 0. Second, two conditions: the total length
        #of the dataset and the percentage of measured temperatures. 
            
                
                
        ds_flux = ds_flux[ds_met.Tair_qc < 1]
        ds_met = ds_met[ds_met.Tair_qc < 1]
        ds_flux = ds_flux[ds_met.SWdown > 1]
        ds_met = ds_met[ds_met.SWdown > 1]
        ds_flux = ds_flux.drop(ds_flux.between_time("23:00", "6:00").index)
        ds_met = ds_met.drop(ds_met.between_time("23:00", "6:00").index)
        
            
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
        
        

        
        #dividing data in different temperature ranges for LH, SH and NEE 
        # average ratio (LH, SH, NEE measurements) for all temperatures
        ratioLH_overall_mean1 = (dftemp_sorted.measuredLH.sum()/
                                dftemp_sorted[0].sum()).round(2)      
        ratioSH_overall_mean1 = (dftemp_sorted.measuredSH.sum()/
                                dftemp_sorted[0].sum()).round(2)
        #not each dataset contains NEE
        if dftemp_sorted.measuredNEE.sum()>1:
            ratioNEE_overall_mean1 = (dftemp_sorted.measuredNEE.sum()/
                                     dftemp_sorted[0].sum()).round(2)
        else: 
            ratioNEE_overall_mean1 = np.nan

        
        #shorten the filename to first 6 letters, which is the abberviation 
        #for the site
        mname = os.path.splitext(os.path.basename(m))[0]
        mname=mname[0:6]

        
        #Adding the results to the empty lists
        results_LH_nohw.append([mname, mean_tair_measured,
                          ratioLH_overall_mean1])                          

        results_SH_nohw.append([mname, mean_tair_measured, 
                          ratioSH_overall_mean1]) 
 
        results_NEE_nohw.append([mname, mean_tair_measured, 
                          ratioNEE_overall_mean1])

       #Making pandas dataframes
df_results_LH_hw = pd.DataFrame(results_LH_hw)
df_results_SH_hw = pd.DataFrame(results_SH_hw)
df_results_NEE_hw = pd.DataFrame(results_NEE_hw)

    #adding names to columns of dataframes
df_results_LH_hw.columns = ['site', 'meantemp','overall']
df_results_SH_hw.columns = ['site', 'meantemp', 'overall']        
df_results_NEE_hw.columns = ['site', 'meantemp', 'overall']

     
 
    #Making pandas dataframes
df_results_LH_nohw = pd.DataFrame(results_LH_nohw)

df_results_SH_nohw = pd.DataFrame(results_SH_nohw)
df_results_NEE_nohw = pd.DataFrame(results_NEE_nohw)

    #adding names to columns of dataframes
df_results_LH_nohw.columns = ['site', 'meantemp',   'overall']

df_results_SH_nohw.columns = ['site', 'meantemp', 'overall']        

df_results_NEE_nohw.columns = ['site', 'meantemp', 'overall']
     
            
        # creating baseplots with zoom  
#overall performance LH, SH, NEE
fig, ax = plt.subplots (nrows = 3, ncols = 2, figsize = (12,15))
p1 = Europe(df_results_LH_hw.overall, "Overall LH measurement ratio heatwave", ax[0,0])
p2 = Europe(df_results_SH_hw.overall, "Overall SH measurement ratio heatwave", ax[1,0])
p3 = Europe(df_results_NEE_hw.overall, "Overall NEE measurement ratio heatwave", ax[2,0])    

# creating baseplots with zoom  
#overall performance LH, SH, NEE
p4 = Europe(df_results_LH_nohw.overall, "Overall LH measurement ratio non-heatwave", ax[0,1])
p5 = Europe(df_results_SH_nohw.overall, "Overall SH measurement ratio non-heatwave", ax[1,1])
p6 = Europe(df_results_NEE_nohw.overall, "Overall NEE measurement ratio non-heatwave", ax[2,1])
plt.tight_layout()


plot_dir = "../plots"
ofname = "Europe.pdf"
fig.savefig(os.path.join(plot_dir, ofname),
            bbox_inches='tight', pad_inches=0.1)
        
        
        
        
        
        
        
