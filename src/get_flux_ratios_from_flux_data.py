#!/usr/bin/env python

"""
I use this script to determine the ratio of measurements of fluxes compared to 
the number of temperature measurements for FLUXNET and LaThuille sites. 

This is done for latent heat, sensible heat and NEE. I focus on 
extreme temperatures (lower and upper 2.2% of the temperature distribution
of each site )


"""

__author__ = "Sophie van der Horst"
__version__ = "1.0 (25.10.2018)"
__email__ = "sophie.vanderhorst@wur.nl"

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import glob
import xarray as xr
import os


def main(files_met, files_flux, ofname1, ofname2, ofname3, ofname4, plot_dir):

    # empty lists to add data in for the table/figures
    results_LH = []
    results_SH = []
    results_NEE = []
    lons = []
    lats = []

    # creating dataframes for barplots of global temperature distribution
    df_temp = pd.DataFrame()
    df_LH = pd.DataFrame()
    df_SH = pd.DataFrame()
    df_NEE = pd.DataFrame()

    for m,f in zip(files_met, files_flux):
        print(m,f)

        ds_met = open_file(m)
        ds_flux = open_file(f)

        lat = ds_met['latitude'].mean()
        lon = ds_met['longitude'].mean()
        
        if len(ds_met) > ((2*365*48)/3) and (len(ds_met[ds_met.Tair_qc < 1])/len(ds_met))>0.5:
                 
            (ds_met, ds_flux,
             Tair_measured, mean_tair_measured) = screen_data(ds_met, ds_flux)
    
            ppt_yearly = ((ds_met.Rainf.mean())*48*365)
            stdev_tair_measured = np.std(Tair_measured.Tair) 
            # Ignore data less than 8 months long ... and athe total length
            # of the dataset and the percentage of measured temperatures.
        

            # Create plot normal distribution of the temperature
            plot_normal_dist(Tair_measured, mean_tair_measured, ds_met, plot_dir)

            # creating bins of 1 K and assigning a temperature to a bin.
            minimum_tair = (min(ds_met.Tair))
            maximum_tair = (max(ds_met.Tair))
            bins = np.arange(np.floor(minimum_tair),
                             np.ceil(maximum_tair+1)).astype(int)
            bin_label = np.arange(np.floor(minimum_tair),
                                  np.ceil(maximum_tair)).astype(int)
            data_binned = pd.cut(ds_met.Tair, bins, labels=bin_label)

            # Adding this temperaturebin to the datasets.
            ds_met.loc[:,"tempbin"] = data_binned
            ds_flux.loc[:, "tempbin"] = data_binned

            # For each bin, count the amount of measurements create a
            # pandas Dataframe for temperature
            temp_sorted = ds_met.groupby(['tempbin']).size()

            # Create a pandas Dataframe for temperature and add binlabel
            dftemp_sorted = pd.DataFrame(temp_sorted)
            dftemp_sorted.loc[:,"bin_label"] = bin_label

            # LATENT HEAT
            # filtering out only the measured latent heats
            ds_met_measuredLH = ds_met[ds_flux.Qle_qc < 1]
            ds_flux_measuredLH = ds_flux[ds_flux.Qle_qc < 1]

            # Ordering the data according to the temperature bin, making a
            # pandaframe and equalizing the index to the index
            #o f the temperature dataframe.
            ds_flux_measuredLH_sorted = ds_flux_measuredLH.groupby(['tempbin']).size()
            df_flux_measuredLH_sorted = pd.DataFrame(ds_flux_measuredLH_sorted)
            dftemp_sorted = dftemp_sorted.set_index(df_flux_measuredLH_sorted.index)
            measuredLH = df_flux_measuredLH_sorted.iloc[:,0]

            # adding the measured LH to temperature pandas dataframe
            dftemp_sorted.loc[:,"measuredLH"]=measuredLH

            # for each temperature bin add: the fraction of LH measurements, the
            # average temperature of the site, stdev
            # of the temperature of the site and the stdev scale.
            dftemp_sorted.loc[:,'ratioLH'] = dftemp_sorted['measuredLH']/\
                                                dftemp_sorted.iloc[:,0]
            dftemp_sorted.loc[:,"averagetemp"] = mean_tair_measured
            dftemp_sorted.loc[:,"stdev"] = stdev_tair_measured
            dftemp_sorted.loc[:,"stdevscale"] = (((dftemp_sorted.loc[:,"bin_label"])-
                                                 (dftemp_sorted.loc[:,"averagetemp"]))/
                                                    (dftemp_sorted.loc[:,"stdev"]))

            #SENSIBLE HEAT
            # filtering out only the measured sensible heats
            ds_met_measuredSH = ds_met[ds_flux.Qh_qc<1]
            ds_flux_measuredSH = ds_flux[ds_flux.Qh_qc<1]

            # Ordering the data according to the temperaturebin, making a
            # pandaframe and equalizing the index to the index of the temperature dataframe.
            ds_flux_measuredSH_sorted = ds_flux_measuredSH.groupby(['tempbin']).size()
            df_flux_measuredSH_sorted = pd.DataFrame(ds_flux_measuredSH_sorted)
            dftemp_sorted = dftemp_sorted.set_index(df_flux_measuredSH_sorted.index)
            measuredSH=df_flux_measuredSH_sorted.iloc[:,0]

            # adding the measured SH to temperature pandas dataframe and the
            # fraction of SH measurements
            dftemp_sorted.loc[:,"measuredSH"]=measuredSH
            dftemp_sorted.loc[:,'ratioSH'] = dftemp_sorted['measuredSH']/\
                                            dftemp_sorted.iloc[:,0]

            #NEE
            # filtering out only the measured NEE
            ds_met_measuredNEE = ds_met[ds_flux.NEE_qc<1]
            ds_flux_measuredNEE = ds_flux[ds_flux.NEE_qc<1]

            # Ordering the data according to the temperature bin, making a
            # pandaframe and equalizing the index to the index of the temperature dataframe
            ds_flux_measuredNEE_sorted = ds_flux_measuredNEE.groupby(['tempbin']).size()
            df_flux_measuredNEE_sorted = pd.DataFrame(ds_flux_measuredNEE_sorted)
            dftemp_sorted = dftemp_sorted.set_index(df_flux_measuredNEE_sorted.index)
            measuredNEE=df_flux_measuredNEE_sorted.iloc[:,0]

            # adding the measured NEE to temperature pandas dataframe and the
            # fraction of NEE measurements
            dftemp_sorted.loc[:,"measuredNEE"] = measuredNEE
            dftemp_sorted.loc[:,'ratioNEE'] = dftemp_sorted['measuredNEE']/\
                                                dftemp_sorted.iloc[:,0]

            # only include temperature bins with >10 temperature measurements
            dftemp_sorted = dftemp_sorted[dftemp_sorted[0] > 10.]

            # creating data for histogram for global temperature
            df_temp = pd.concat([df_temp, dftemp_sorted.iloc[:,0]], axis=1)
            df_LH = pd.concat([df_LH, dftemp_sorted['measuredLH']], axis=1)
            df_SH = pd.concat([df_SH, dftemp_sorted['measuredSH']], axis=1)
            df_NEE = pd.concat([df_NEE, dftemp_sorted['measuredNEE']], axis=1)

            # Working with the energy balance. Only measured SH, LH and Rnet.
            # This was done over a long period of time.
            ds_flux_energybalance = ds_flux[ds_flux.Qle_qc < 1]
            ds_flux_energybalance = ds_flux_energybalance[ds_flux.Qh_qc < 1]

            # Explain why I'm doing this ...?

            # not every dataset contained a Rnet.
            if 'Rnet' in ds_flux_energybalance.columns:
                ds_flux_energybalance_Rnet = ds_flux_energybalance[ds_flux.Rnet_qc <1]
                mean_Rnet = ds_flux_energybalance_Rnet.Rnet.mean()
                mean_LH = ds_flux_energybalance_Rnet.Qle.mean()
                mean_SH = ds_flux_energybalance_Rnet.Qh.mean()

                # Calculating the residual energy
                energybalance = mean_Rnet - mean_LH - mean_SH

            # dividing data in different temperature ranges for LH, SH and NEE
            # average ratio (LH, SH, NEE measurements) for all temperatures
            ratioLH_overall_mean = (dftemp_sorted.measuredLH.sum()/
                                    dftemp_sorted[0].sum()).round(2)
            ratioSH_overall_mean = (dftemp_sorted.measuredSH.sum()/
                                    dftemp_sorted[0].sum()).round(2)

            # not each dataset contains NEE
            if dftemp_sorted.measuredNEE.sum()>1:
                ratioNEE_overall_mean = (dftemp_sorted.measuredNEE.sum()/
                                         dftemp_sorted[0].sum()).round(2)
            else:
                ratioNEE_overall_mean = np.nan

            # This part is for temperature ranges I am not using in my study yet:
            # average ratio (LH, SH, NEE measurements) if temperature is < mean
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

            # average ratio (LH, SH, NEE measurements) if temperature is > mean
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

            # average ratio (LH, SH, NEE measurements) if temperature within
            # 1 stdev from the mean
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

            # average ratio (LH, SH, NEE measurements) if temperature is > 1SD
            # away from the mean
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

            # average ratio (LH, SH, NEE measurements) if temperature is < 1stdev
            # away from the mean
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

            # average ratio (LH, SH, NEE measurements) if temperature is < 2stdev
            # away from the mean
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

            # shorten the filename to first 6 letters, which is the abberviation
            # for the site
            mname = os.path.splitext(os.path.basename(m))[0]
            mname = mname[0:6]

            # vegetation
            vegetation = ds_met['IGBP_veg_long'].str.decode("utf-8")

            # colours for the scatterplot
            if mean_tair_measured >= 295:
                color1 = 'Firebrick'
            elif mean_tair_measured <275:
                color1 = 'blue'
            else:
                color1 = 'black'

            # Adding the results to the empty lists
            results_LH.append([mname, lat, lon, mean_tair_measured,  ppt_yearly,
                              ratioLH_overall_mean, ratioLH_smallerthan_mean, ratioLH_largerthan_mean,
                              ratioLH_within1_stdev_mean, ratioLH_smallerthan1_stdev_mean,
                              ratioLH_largerthan1_stdev_mean, ratioLH_smallerthan2_stdev_mean,
                              ratioLH_largerthan2_stdev_mean, color1, energybalance, vegetation[0]])
            results_SH.append([mname, lat, lon, mean_tair_measured, ppt_yearly,
                              ratioSH_overall_mean, ratioSH_smallerthan_mean, ratioSH_largerthan_mean,
                              ratioSH_within1_stdev_mean, ratioSH_smallerthan1_stdev_mean,
                              ratioSH_largerthan1_stdev_mean,ratioSH_smallerthan2_stdev_mean,
                              ratioSH_largerthan2_stdev_mean, color1, energybalance, vegetation[0]])
            results_NEE.append([mname, lat, lon, mean_tair_measured, ppt_yearly,
                              ratioNEE_overall_mean, ratioNEE_smallerthan_mean, ratioNEE_largerthan_mean,
                              ratioNEE_within1_stdev_mean, ratioNEE_smallerthan1_stdev_mean,
                              ratioNEE_largerthan1_stdev_mean, ratioNEE_smallerthan2_stdev_mean,
                              ratioNEE_largerthan2_stdev_mean, color1, energybalance, vegetation[0]])

            # Making pandas dataframes
            df_results_LH = pd.DataFrame(results_LH)
            df_results_SH = pd.DataFrame(results_SH)
            df_results_NEE = pd.DataFrame(results_NEE)

            # adding names to columns of dataframes
            df_results_LH.columns = ['site', 'lat', 'lon',  'meantemp', 'meanprec',   'overall',
                                    'smallerthanmean', 'largerthanmean', 'within1stdev',
                                    'smallerthan1stde', 'largerthan1stdev',
                                    'smallerthan2stdev', 'largerthan2stdev',
                                    'color', 'energybalance', 'vegetation']
            df_results_SH.columns = ['site', 'lat', 'lon', 'meantemp', 'meanprec', 'overall',
                                    'smallerthanmean', 'largerthanmean', 'within1stdev',
                                    'smallerthan1stdev', 'largerthan1stdev',
                                    'smallerthan2stdev', 'largerthan2stdev',
                                    'color', 'energybalance', 'vegetation']
            df_results_NEE.columns = ['site', 'lat', 'lon', 'meantemp', 'meanprec',  'overall',
                                     'smallerthanmean', 'largerthanmean', 'within1stdev',
                                     'smallerthan1stdev', 'largerthan1stdev',
                                     'smallerthan2stdev', 'largerthan2stdev', 'color',
                                     'energybalance', 'vegetation']
            ####
            # Add to above ...
            ###
            #latitudes and longitudes
            lat = ds_met['latitude'].mean()
            lon = ds_met['longitude'].mean()
            lats.append(lat)
            lons.append(lon)
            
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
                
    
    df_results_LH.to_csv(ofname1)
    df_results_SH.to_csv(ofname2)
    df_results_NEE.to_csv(ofname3)
    df_total.to_csv(ofname4)
            
    


def open_file(fname):
    ds = xr.open_dataset(fname)
    ds = ds.squeeze(dim=["x","y"], drop=True).to_dataframe()
    ds = ds.reset_index()
    ds = ds.set_index('time')
    return (ds)

def screen_data(ds_met, ds_flux):

    #Does the data have enough recorded temperatures? First, selecting
    #only measurements with qc = 0.
    Tair_measured = ds_met[ds_met.Tair_qc < 1]
    mean_tair_measured = np.mean(Tair_measured.Tair)

    # Filter to measured temperature (qc=0) and only shortwave
    #incoming radiation >0 and filtering out data between 11 pm and 6 am
    ds_flux = ds_flux[ds_met.Tair_qc < 1]
    ds_met = ds_met[ds_met.Tair_qc < 1]
    ds_flux = ds_flux[ds_met.SWdown > 1]
    ds_met = ds_met[ds_met.SWdown > 1]
    ds_flux = ds_flux.drop(ds_flux.between_time("23:00", "6:00").index)
    ds_met = ds_met.drop(ds_met.between_time("23:00", "6:00").index)

    return (ds_met, ds_flux, Tair_measured, mean_tair_measured)

def plot_normal_dist(Tair_measured, mean_tair_measured,  ds_met, plot_dir):
    # Create plot normal distribution of the temperature

    width = 9
    height = 6
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=0.05)
    fig.subplots_adjust(wspace=0.05)
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.family'] = "sans-serif"
    plt.rcParams['font.sans-serif'] = "Helvetica"
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['font.size'] = 14
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14

    ax = fig.add_subplot(111)

    stdev_tair_measured = np.std(Tair_measured.Tair)
    x = np.linspace(min(ds_met.Tair), max(ds_met.Tair), 1000)
    normal_pdf = norm.pdf(x, mean_tair_measured, stdev_tair_measured)

    ax.plot(x, normal_pdf, "blue", linewidth=2)
    ax.set_title("Temperature distribution")
    ax.set_xlabel("Temperature ($^\circ$C)")
    ax.set_ylabel("Frequency")

    ofname = "normal_distribution.pdf"
    fig.savefig(os.path.join(plot_dir, ofname),
                bbox_inches='tight', pad_inches=0.1)

if __name__ == "__main__":

    files_met = sorted(glob.glob("../data/Met/*"))
    files_flux = sorted(glob.glob("../data/Flux/*"))
    out_path = "../data/processed"
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    plot_dir = "../plots"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    ofname1 = os.path.join(out_path, "results_LH.csv")
    ofname2 = os.path.join(out_path, "results_SH.csv")
    ofname3 = os.path.join(out_path, "results_NEE.csv")
    ofname4 = os.path.join(out_path, "globaltemp.csv")

    main(files_met, files_flux, ofname1, ofname2, ofname3, ofname4, plot_dir)
