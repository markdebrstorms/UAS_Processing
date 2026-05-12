# -*- coding: utf-8 -*-
"""
Created on Mon May 11 21:29:20 2026

@author: mdebruin2
"""

import numpy as np
import datetime
from metpy.interpolate import interpolate_to_grid, remove_nan_observations
import math
from metpy.calc import mixing_ratio_from_relative_humidity, potential_temperature
from metpy.calc import dewpoint_from_relative_humidity, equivalent_potential_temperature, wind_direction, wind_components, wind_speed
from metpy.units import units
import pandas as pd
import os
import xarray as xr
import re
from PIL import Image
from scipy.interpolate import Rbf, griddata, interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tkinter as tk
from tkinter import filedialog
from boundaryCoords import *
from mesoCoords import *
from radarcalc import *

def MHP_time_response(var_hat, t, tau):
    # Time response due to MHP sensor lag. Needs to be applied
    # to temp and RH. Equation is Var = tau*(dvar/dt)+ var_hat
    # var = corrected temp or RH
    # tau = time response
    # dvar/dt = centered differencing (using previous measurement and next measurement)
    # var_hat = actual observed measurement
      
    var = []
    for i in range(1,np.asarray(var_hat).shape[0] - 1):
        t_diff = (t[i+1] - t[i-1]).total_seconds()
        var_new = (tau*((var_hat[i+1]-var_hat[i-1])/(1*t_diff))) + var_hat[i]
        var.append(var_new)
                                      
    var_corr = np.asarray(var)
    time_corr = t[1:-1]
    return var_corr, time_corr

def iPRH_time_response(var_hat, t, tau):
    # Time response due to iPTH RHtemp sensor lag. Needs to be applied
    # to temp. Equation is Var = tau*(dvar/dt)+ var_hat
    # var = corrected temp or RH
    # tau = time response
    # dvar/dt = centered differencing (using previous measurement and next measurement)
    # var_hat = actual observed measurement
    
    var = []
    for i in range(1,np.asarray(var_hat).shape[0] - 1):
        t_diff = (t[i+1] - t[i-1]).total_seconds()
        var_new = (tau*((var_hat[i+1]-var_hat[i-1])/(1*t_diff))) + var_hat[i]
        var.append(var_new)
                                      
    var_corr = np.asarray(var)
    time_corr = t[1:-1]
    return var_corr, time_corr

def iprhCorrect(rh, temp):
    rhcA = np.asarray([])
    for r, t in zip(rh, temp):
        rhc = (13.68 + (0.1244*r) + (-0.03791*t) + (0.0334*(r**2)) + (0.001814*r*t) + (-0.0003154*(r**3)) + (-6.892e-5*(r**2)*t) +
        (4.864e-7*(r**4)) + (5.87e-7*(r**3)*t))
        #rhc = -49.4041 + (1.2262*r) + (0.3387*t) + (-7.8942e-4*(r**2)) + (-1.3310e-04*r*t) + (-6.2320e-04*(t**2))
        rhcA = np.append(rhcA, rhc)
    rhcA = np.clip(rhcA, 0, 99.9)
    return rhcA

# Filters out suspect comet wind data by deriving vehicle velocity and acceleration
# Fast accelerations > 2m/s2  and 2 degrees/s are filtered out using a 0/1 mask
def cometQCwind(cUTC, clat, clon, cvdir):
    # Use point to point distance and time change to derive both speed and acceleration
    speedA = np.asarray([np.nan])
    speedDirA = np.asarray([np.nan])
    for idx, t in enumerate(cUTC[1:-1]):
        elapsed = (cUTC[idx+1] - cUTC[idx-1]).total_seconds()
        dist = getDistance(clat[idx+1], clon[idx+1], clat[idx-1], clon[idx-1])
        speedA = np.append(speedA, np.abs(dist*1000)/np.abs(elapsed))
        speedDirA = np.append(speedDirA, np.abs(cvdir[idx+1] - cvdir[idx-1])/np.abs(elapsed))
    speedA = np.append(speedA, np.nan)
    accelA = np.asarray([np.nan])
    accelDirA = np.asarray([np.nan])
    for idx, s in enumerate(speedA[1:-1]):
        accel = (speedA[idx+1] - speedA[idx-1])
        accelA = np.append(accelA, accel)
        accelDir = (speedDirA[idx+1] - speedDirA[idx-1])
        accelDirA = np.append(accelDirA, accelDir)
    accelA = np.append(accelA, np.nan)
    accelDirA = np.append(accelDirA, np.nan)
    
    # Filter values
    at = 2
    dt = 2
    filtL = []
    for idx, i in enumerate(cUTC):
        # Values larger than 170 are assumed to be caused by the vehicle
        # direction switching between 0 and 360. This is definitely a bit subjective
        if abs(accelA[idx]) > at or 170 > abs(accelDirA[idx]) > dt or np.isnan(abs(accelA[idx])) or np.isnan(abs(accelDirA[idx])):
            filtL.append(1)
            print(abs(accelA[idx]), abs(accelDirA[idx]))
        else:
            filtL.append(0)
            
    return filtL

# Filters out suspect comet thermodynamic data by deriving vehicle relative flow
# Derivation works by projecting the wind vector to the comet motion and subtracting
# Values between -5 and 0 are treated as bad, with negative VRF being rear-to-front flow
def cometQCthermo(cu, cv, cUTC, clat, clon):
    # Use point to point distance and time change to derive speed
    vrfA = np.asarray([np.nan])
    for idx, t in enumerate(cUTC[1:-1]):
        elapsed = (cUTC[idx+1] - cUTC[idx-1]).total_seconds()
        distX = getDistance(clat[idx-1], clon[idx+1], clat[idx-1], clon[idx-1])
        distY = getDistance(clat[idx+1], clon[idx-1], clat[idx-1], clon[idx-1])
        if clat[idx-1] > clat[idx+1]:
            distY = distY * -1
        if clon[idx-1] > clon[idx+1]:
            distX = distX * -1
        speedX = (distX*1000)/np.abs(elapsed)
        speedY = (distY*1000)/np.abs(elapsed)
        
        # This is a vector projection. The observed wind is projected to the
        # comet motion vector. The vrf is the difference between comet motion
        # and projected wind vector. The resulting vector can point down the
        # comet motion vector or against it. The cosine similarity block determines
        # if it is rear-to-front flow (given a negative value), or front-to-rear
        # (given a posive value).
        num = (cu[idx] * speedX) + (cv[idx] * speedY)
        denom = (speedX**2) + (speedY**2)
        coeff = num / denom
        windProj = (coeff * speedX, coeff * speedY)
        vrf = (speedX - windProj[0], speedY - windProj[1])
        vrfmag = math.sqrt((vrf[0]**2) + (vrf[1]**2))
        
        # Silly quirky vector math to determine which direction the vectors
        # are pointing (cosine similarity)
        if np.isclose(np.dot(vrf, (speedX, speedY)) / (np.linalg.norm(vrf) * np.linalg.norm((speedX, speedY))), 1.0, atol=1e-4):
            vrfmag = vrfmag * 1
        elif np.isclose(np.dot(vrf, (speedX, speedY)) / (np.linalg.norm(vrf) * np.linalg.norm((speedX, speedY))), -1.0, atol=1e-4):
            vrfmag = vrfmag * -1     
        else:
            # Note that this should always (to my knowledge) be triggered by
            # nan values being input.
            print('1/2 sign mismatches, nan value likely encountered')
        vrfA = np.append(vrfA, vrfmag)
    vrfA = np.append(vrfA, np.nan)
    
    # Filter values
    vrfF = (-5, 0)
    filtL = []
    for idx, i in enumerate(cUTC):
        if vrfF[1] > vrfA[idx] > vrfF[0] or np.isnan(abs(vrfA[idx])):
            filtL.append(1)
        else:
            filtL.append(0)
            
    return filtL

# Derives theta_v given arrays (magnitude only) of temp, pressure, and RH (in %)
# Returns metpy array w/ units 
def thetav(t, p, rh):
    wkg = mixing_ratio_from_relative_humidity(p* units.hPa, t*units.degC, np.array(rh)/100)
    theta = potential_temperature(p*units.hPa, t*units.degC)
    tv = theta*(1+(0.61*wkg))
    return tv

# Derives theta given arrays (magnitude only) of temp, pressure
# Returns metpy array w/ units 
def theta(t, p):
    theta = potential_temperature(p*units.hPa, t*units.degC)
    return theta

# Derives theta_e given arrays (magnitude only) of temp, pressure, and RH (in %)
# Returns metpy array w/ units 
def thetaE(t, p, rh):
    dp = dewpoint_from_relative_humidity(t*units.degC, np.array(rh)/100)
    thetae = equivalent_potential_temperature(p* units.hPa, t*units.degC, dp)
    return thetae

# Given an xarray dataset, returns list of datetime objects of UAS sample times
# This is for the old 2019 format that has since been changed
def UTC19(ds, n):
    lftimes = ds['mhp_gps_time'][0][None:None:n]
    startTime = datetime.datetime(1980,1,6,0,0,0)
    UTCtimeList = [startTime + datetime.timedelta(seconds=float(timeIndex.values) / 1000000000) for timeIndex in lftimes]
    return UTCtimeList

# Given an xarray dataset, returns list of datetime objects of UAS sample times
# This is for the old 2019 format that has since been changed
def UTCrs(ds, n):
    lftimes = ds['vn_gps_time'][0][None:None:n]
    startTime = datetime.datetime(1980,1,6,0,0,0)
    UTCtimeList = [startTime + datetime.timedelta(seconds=float(timeIndex.values) / 1000000000) for timeIndex in lftimes]
    return UTCtimeList

# Given an xarray dataset, returns list of datetime objects of UAS sample times
# This is for the old 2019 format that has since been changed
def UTCmhp(ds, n):
    lftimes = ds['mhp_gps_time'][0][None:None:n]
    startTime = datetime.datetime(1980,1,6,0,0,0)
    UTCtimeList = [startTime + datetime.timedelta(seconds=float(timeIndex.values) / 1000000000) for timeIndex in lftimes]
    return UTCtimeList

# Given an xarray dataset, returns list of datetime objects of UAS sample times
# This will work for the new B1 files as of august 2025
# Name isn't accurate, after reformatting the B1 files, this function works for 2019/2023
def UTC23(ds, n):
    times = ds.time_10hz[None:None:n]
    startTimeList = times.Units.split(' ')[2::]
    timeString = startTimeList[0]+' '+startTimeList[1]
    startTime = datetime.datetime.strptime(timeString, '%Y-%m-%d %H:%M:%S')
    UTCtimeList = [startTime + datetime.timedelta(seconds=float(timeIndex.values)) for timeIndex in times]
    return UTCtimeList

# Given an xarray dataset, returns list of datetime objects of comet sample times
def UTCcom(ds, n):
    times = ds.time[None:None:n]
    startTime = startTime = datetime.datetime(1970,1,1,0,0,0)
    UTCcom = [startTime + datetime.timedelta(seconds=float(timeIndex.values) / 1000000000) for timeIndex in times]
    return UTCcom

# Given an xarray dataset, returns list of datetime objects of sounding sample times
# Works for the 2019 .csv files
def sondeUTC(df):
    times = df['Date-Time'].values
    sondeUTC = [datetime.datetime.strptime(time[1:17], '%Y-%m-%dT%H:%M') for time in times]
    return sondeUTC

# Given an xarray dataset, returns list of datetime objects of sounding sample times
# Works for the 2023 .csv files
def sondeUTC23(file):
    df = pd.read_csv(file, skiprows=[1], encoding='unicode_escape')
    launch = datetime.datetime.strptime(file[-20:-5], '%Y_%m_%d_%H%M')
    times = df['Elapsed'].values
    sondeUTC = [launch + datetime.timedelta(seconds=float(time)) for time in times]
    return sondeUTC, df

# This one is weird but important for my thesis
# The premise of this function is that it will grab a slice of the dataset
# So instead of deriving every variable for the entire transect every time I
# run my scripts, only the time interval that is needed is used.
#
# It's weirder because I merged this function with my hysteresis correction
# I don't remember why, but there was a reason.
#
# You could just derive all the variables for the whole deployment and save them
# and save yourself some extra coding and forget this function. But this way is
# more efficient for constantly changing methodology.
def lfPlotables(ds, UTCtimeList, startTime, endTime, var, n, tau):
    idxList = []
    valUTC = []
    # Find indices within the provided time range
    for idx, time in enumerate(UTCtimeList):
        if time > startTime and time < endTime:
            idxList.append(idx)
            valUTC.append(time)
            
    # If a correction is going to be applied, grabs an extra index from start and end 
    if var in ['air_rh_temp_ipth', 'air_temperature', 'relative_humidity', 'air_rh_ipth']:
        valUTC = [UTCtimeList[idxList[0]-1]] + valUTC + [UTCtimeList[idxList[-1]+1]]
    try:
        idxSlice = slice(idxList[0], idxList[-1]+1)
        lfvars = ds[var][None:None:n].isel(time=slice(idxList[0], idxList[-1]+1)).values
        
        # Applies hysteresis correction to certain variables
        # If not applying correction, don't just set tau=0, delete the variable
        # from the list.
        if var in ['air_rh_temp_ipth', 'air_temperature', 'relative_humidity', 'air_rh_ipth']:
            lfvars = ds[var][None:None:n].isel(time=slice(idxList[0]-1, idxList[-1]+2)).values
            
            if startTime.year == 2023:
                # Smooths raw values first
                lfvarsS = gaussianTime(valUTC, lfvars, tau, 0.01)
                # Applies the correction
                lfvars, abrevTime = iPRH_time_response(lfvarsS, valUTC, tau)
            else:
                # Correction was slightly different for MHP, but now that I think
                # about it the functions are the same now, this if/else block
                # shouldn't be needed
                lfvarsS = gaussianTime(valUTC, lfvars, tau, 0.01)
                lfvars, abrevTime = MHP_time_response(lfvarsS, valUTC, tau)
        else:
            lfvars = ds[var][None:None:n].isel(time=slice(idxList[0], idxList[-1]+1)).values
        return lfvars, valUTC
            
    except Exception as error:
        print(error)
        return None, None

# Same idea as lfPlotables, a slice of the comet dataset is returned rather than
# the whole deployment.    
def comPlotables(ds, UTCtimeList, startTime, endTime, var, n):
    idxList = []
    timeList = []
    for idx, time in enumerate(UTCtimeList):
        if time > startTime and time < endTime:
            idxList.append(idx)
            timeList.append(time)
    try:
        data = ds[var][None:None:n].isel(time_dim=slice(idxList[0], idxList[-1]+1)).values
    except Exception as error:
        print('Comet data error:')
        print(error)
        return None, None
    return data, timeList

# Returns a list of variable, time values within a time range
def timeRange(ds, start, end, UTC):
    dsR = [i for idx, i in enumerate(ds) if UTC[idx] > start and UTC[idx] < end]
    utcR = [i for idx, i in enumerate(UTC) if i > start and i < end]
    return dsR, utcR

# I don't remember what this does
# Eliminates duplicates of something
def dupMask(vals):
    seen = {}
    for index, value in enumerate(vals):
        if value not in seen:
            seen[value] = index  # Mark as not unique if it appears again

    unique_indices = [index for index in seen.values() if index is not None]
    
    return unique_indices

# NSSL Probe datasets are different than comets
# This returns variables formatted according to comet variables
def p2format(infodf):
    filepath = os.path.join(os.getcwd(), str(infodf['Directory'][0]), infodf['Comet'][0])
    nameList = ['Platform', 'Time', 'Lat', 'Lon', 'Alt', 'Tfast', 'Tslow', 'RH', 'Pres', 'Dir', 'Spd', 'qc1', 'qc2', 'qc3', 'qc4']
    df = pd.read_csv(filepath, delimiter='\s+', names=nameList, header=None)
    
    times = df['Time'].values
    dupIdx = dupMask(times)
    times = np.asarray(times)[dupIdx]
    t0 = datetime.datetime(2019, 5, 17, 0, 0, 0)
    newTimes = []
    for time in times:
        hours = int(time)
        minutes = (time*60) % 60
        seconds = (time*3600) % 60
        t1 = t0 + datetime.timedelta(hours=hours, minutes=minutes, seconds=seconds)
        newTimes.append(t1)
        
    lat = df['Lat'].values[dupIdx]
    lon = df['Lon'].values[dupIdx]
    alt = df['Alt'].values[dupIdx]
    temp = df['Tslow'].values[dupIdx]
    rh = df['RH'].values[dupIdx]
    pres = df['Pres'].values[dupIdx]
    wdir = df['Dir'].values[dupIdx]
    wspd = df['Spd'].values[dupIdx]
    
    tv = thetav(temp, pres, rh)
    te = thetae(temp, pres, rh)
    
    return newTimes, lat, lon, alt, temp, rh, pres, wdir, wspd, tv.magnitude, te.magnitude

# Built for parsing TORUS-LItE comet text files, not the netcdf datasets
def comParse(infodf, caseID, start, end):
    filepath = os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), infodf['Comet'][caseID])
    nameList = ['Date', 'Time', 'Lat', 'Lon', 'Alt', 'Pressure', 'Temp fast C', 'Temp slow C', 'Logger RH', 'Calculated Corrected RH',
                'Dewpoint C', 'Mixing Ratio g/kg', 'Theta K', 'Theta-V K', 'Theta-E K', 'Wind speed m/s', 'Wind direction', 'Fluxgate heading',
                'Vehicle speed m/s', 'Computer Time epoch', 'Logger time']
    skip = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    df = pd.read_csv(filepath, delimiter=',', names=nameList, header=None, skiprows=skip, dtype='str')
    date = df["Date"]
    time = df["Time"]
    cspd = np.asarray(df['Wind speed m/s'], dtype=float)
    cdir = np.asarray(df['Wind direction'], dtype=float)
    calt = np.asarray(df['Alt'], dtype=float)
    clat = np.asarray(df['Lat'], dtype=float)
    clon = np.asarray(df['Lon'], dtype=float)
    ctv = np.asarray(df['Theta-V K'], dtype=float)
    cte = np.asarray(df['Theta-E K'], dtype=float)
    cpres = np.asarray(df['Pressure'], dtype=float)
    dtL = []
    idxL = []
    for idx, d in enumerate(date):
        t = time[idx]
        # Don't remember how this datetime formatting works honestly
        dt = datetime.datetime(2000+int(d[4:6]), int(d[2:4]),int(d[0:2]), int(t[0:2]), int(t[2:4]), int(t[4:6]))
        if dt >= start and dt <= end:
            idxL.append(True)
            dtL.append(dt)
        else:
            idxL.append(False)
    cu, cv = winduv(cspd, cdir)
    
    return dtL, clat[idxL], clon[idxL], cu[idxL], cv[idxL], calt[idxL], ctv[idxL], cte[idxL], cpres[idxL]
    
# Some function for skew-T plotting, finds what the plot temp bounds should be
# for a given altitude
def tempBounds(tempList, altList, alt):
    tList = []
    for hidx, temps in enumerate(tempList):
        subAlt = [temp for idx,temp in enumerate(temps) if altList[hidx][idx] <=alt]
        tmin = np.nanmin(subAlt)
        tmax = np.nanmax(subAlt)
        tList.append(tmin)
        tList.append(tmax)
    return (min(tList), max(tList))

# I think this is just a better written version of tempBounds()
def varBounds(var, altList, alt):
    tList = []
    subAlt = [val for idx,val in enumerate(var) if altList[idx] <=alt]
    tmin = np.nanmin(subAlt)
    tmax = np.nanmax(subAlt)
    tList.append(tmin)
    tList.append(tmax)
    return [min(tList), max(tList)]

# Older UAS files had erroneous coordinate values occasionally
def realCoordCheck(lats, lons, varList):
    goodIdx = []
    goodLat = []
    goodLon = []
    for idx, lat in enumerate(lats):
        lon = lons[idx]
        if lat > 5 and lat < 50 and lon > -140 and lon < -70:
            goodIdx.append(idx)
            goodLat.append(lats[idx])
            goodLon.append(lons[idx])
            
    newvarList = []
    for var in varList:
        newVar = [var[idx] for idx in goodIdx]
        newvarList.append(newVar)
    
    return goodLat, goodLon, newvarList

# Given a reference profile and arrays (lists?) of values/altitudes, finds the
# value of at that height on the reference profile and returns the deficit (perturbation)
# Note that the reference profile may need to be in a weird unflattened numpy array
# format. I should have changed that and never felt like it.
def deficit(var, alt, refVar, refAlt):
    minRef = min(refAlt)
    maxRef = max(refAlt)
    tempRg = []
    altRg = []
    idxList = []
    warn = 'All reference elevations met'
    for idx, val in enumerate(alt):
        if val > minRef and val < maxRef:
            tempRg.append(var[idx])
            altRg.append(val)
            idxList.append(idx)
        elif val >= maxRef and val < maxRef+30:
            tempRg.append(var[idx])
            altRg.append(val)
            idxList.append(idx)

    rnd = [math.floor(val[0]) for val in refAlt]
    deficit = []
    for idx, a in enumerate(altRg):
        try:
            Tidx = rnd.index(round(a))
            defi = tempRg[idx] - refVar[Tidx][0]
            deficit.append(defi)
        except:
            Tidx = rnd.index(max(rnd))
            defi = tempRg[idx] - refVar[Tidx][0]
            deficit.append(defi)
            warn = 'Error in elevation matches, max elevation used'
    print(warn)
    return idxList, deficit

# Similar to deficit but returns the values instead of the deficit
def deficitVar(var, alt, refVar, refAlt):
    minRef = min(refAlt)
    maxRef = max(refAlt)
    tempRg = []
    altRg = []
    idxList = []
    warn = 'All reference elevations met'
    for idx, val in enumerate(alt):
        if val > minRef and val < maxRef:
            tempRg.append(var[idx])
            altRg.append(val)
            idxList.append(idx)
        elif val >= maxRef and val < maxRef+30:
            tempRg.append(var[idx])
            altRg.append(val)
            idxList.append(idx)
            
    rnd = [math.floor(val[0]) for val in refAlt]
    newVar = []
    matchRef = []
    matchRefAlt = []
    for idx, a in enumerate(altRg):
        try:
            Tidx = rnd.index(round(a))
            newVar.append(tempRg[idx])
            matchRef.append(refVar[Tidx][0])
            matchRefAlt.append(refAlt[Tidx][0])
        except:
            Tidx = rnd.index(max(rnd))
            newVar.append(tempRg[idx])
            matchRef.append(refVar[Tidx][0])
            matchRefAlt.append(refAlt[Tidx][0])
            warn = 'Error in elevation matches, max elevation used'
    print(warn)
    return idxList, newVar, matchRef, altRg, matchRefAlt

# Plots a profile given a nested list of variables, altitudes, and list of labels
def simpleProfilePlot(varList, alt, labelList, caseID):
    fig, axes = plt.subplots(1, len(varList), figsize=(15,10), facecolor ='white', clear=True)
    plt.rcParams.update({'font.size': 20})
    plt.suptitle(f'Case {caseID} Reference Profile')
    ylim = [np.nanmin(alt), np.nanmin(alt)+600]
    axes[0].set_ylabel('Above Sea Level (m)')
    for idx, ax in enumerate(axes):
        ax.scatter(x=varList[idx], y=alt)
        ax.set_ylim(ylim)
        xbounds = varBounds(varList[idx], alt, np.nanmin(alt)+600)
        ax.set_xlim(xbounds)
        ax.set_xlabel(labelList[idx])
        #ax.axvline(839)
        ax.axhline(1520)
    plt.show()

# Same as previous but with extra finesse    
def complexProfilePlot(varList, alt, labelList, title, caseID):
    fig, ax = plt.subplots(1, 1, figsize=(10,10), facecolor ='white', clear=True)
    plt.rcParams.update({'font.size': 20})
    plt.suptitle(f'Case {caseID} Reference Profile '+title)
    ylim = [np.nanmin(alt[0]), np.nanmin(alt[0])+700]
    ax.set_ylabel('Above Sea Level (m)')
    xmin = 10000
    xmax = 0
    sizes = [15, 15, 100]
    sizes = [15, 15, 15, 15, 15, 15, 15, 100]
    for idx, var in enumerate(varList):
        ax.scatter(x=var, y=alt[idx], zorder=idx, s=sizes[idx])
        ax.set_ylim(ylim)
        xbounds = varBounds(var, alt[idx], np.nanmin(alt[idx])+600)
        if xbounds[0]<xmin:
            xmin = xbounds[0]
        if xbounds[1]>xmax:
            xmax = xbounds[1]
    ax.set_xlim([xmin, xmax])
    ax.set_xlabel('Variable')
    ax.legend(labelList)
    plt.show()

# Manages different filetypes for far field, LIDAR, and NI UAS, returning
# the same variables for each
def profileManager19(infodf, caseID, minAlt):
    file = infodf['Use_Profile'][caseID]
    caseDir = infodf['Directory'][caseID]
    if file[0:3]=='Far':
        df = pd.read_csv(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), 'Far Field', file), skiprows=[0,1])
        temp = df['Filtered Temperature (K)'].values
        temp = temp - 273.15
        rh = df['Filtered Humidity'].values
        p = df['Filtered Pressure (mb)'].values
        alt = df['Filtered Altitude (m)'].values
        angle = df['Filtered Wind Dir'].values
        speed = df['Filtered Wind Spd (m/s)'].values
    elif file[0:3]=='LID':
        df = pd.read_csv(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), 'Far Field', 'LIDAR',file), skiprows=[0,1])
        temp = df['Filtered Temperature (K)'].values
        temp = temp - 273.15
        rh = df['Filtered Humidity'].values
        p = df['Filtered Pressure (mb)'].values
        alt = df['Filtered Altitude (m)'].values
        angle = df['Filtered Wind Dir'].values
        speed = df['Filtered Wind Spd (m/s)'].values
    elif file[0:2]=='NI':
        n = 90
        ni = xr.open_dataset(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), file))
        alt = ni.mhp_vn_alt[0][None:None:n].values
        temp = ni.mhp_air_temp[0][None:None:n].values
        pres = ni.mhp_static_press[0][None:None:n].values
        p = pres[1:-1]
        alt = alt[1:-1]
        rh = ni.mhp_rh[0][None:None:n].values
        niUTC = UTC19(ni, n)
        temp, lfUTC2 = MHP_time_response(temp, niUTC, 11)
        rh, lfUTC = MHP_time_response(rh, niUTC, 11)
        niu = ni.mhp_RSYaw_u_wind[0][None:None:n].values
        niv = ni.mhp_RSYaw_v_wind[0][None:None:n].values
        angle = wind_direction(niu*units('m/s'), niv*units('m/s')).magnitude
        speed = wind_speed(niu*units('m/s'), niv*units('m/s')).magnitude      
    else:
        print('Unrecognized profile filetype')
    
    if minAlt < np.nanmin(alt):
        #Extend profile to the surface
        slp, altN = extendProfile(p, alt, temp, minAlt)
        nans = np.full(len(altN), np.nan)
        temp = np.insert(temp, 0, nans)
        rh = np.insert(rh, 0, nans)
        p = np.insert(p, 0, slp)
        alt = np.insert(alt, 0, altN)
        angle = np.insert(angle, 0, nans)
        speed = np.insert(speed, 0, nans)
    
    simpleProfilePlot([temp, rh, p], alt, ['Temp', 'RH', 'Pressure'], caseID)
    
    return temp, rh, p, alt, speed, angle

# Same as profileManager() but only for wind variables
def profileManager19wind(infodf, caseID):
    file = infodf['Use_Profile'][caseID]
    caseDir = infodf['Directory'][caseID]
    if file[0:3]=='Far':
        df = pd.read_csv(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), 'Far Field', file), skiprows=[0,1])
        angle = df['Filtered Wind Dir'].values
        speed = df['Filtered Wind Spd (m/s)'].values
        alt = df['Filtered Altitude (m)'].values
        return speed, angle, alt
    elif file[0:3]=='LID':
        df = pd.read_csv(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), 'Far Field', 'LIDAR',file), skiprows=[0,1])
        angle = df['Filtered Wind Dir'].values
        speed = df['Filtered Wind Spd (m/s)'].values
        alt = df['Filtered Altitude (m)'].values
        return speed, angle, alt
    elif file[0:2]=='NI':
        n = 90
        ni = xr.open_dataset(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), file))
        alt = ni.mhp_vn_alt[0][None:None:n].values
        niu = ni.mhp_RSYaw_u_wind[0][None:None:n].values
        niv = ni.mhp_RSYaw_v_wind[0][None:None:n].values
        angle = wind_direction(niu*units('m/s'), niv*units('m/s'))
        speed = wind_speed(niu*units('m/s'), niv*units('m/s'))
        return speed.magnitude, angle.magnitude, alt
    else:
        print('Unrecognized profile filetype')

# Manages far field and near inflow profiles present in 2023
# Will not work for new B1 files
def profileManager23(infodf, caseID, minAlt):
    file = infodf['Use_Profile'][caseID]
    caseDir = infodf['Directory'][caseID]
    if file[0:4]=='2023':
        df = pd.read_csv(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), 'Far Field', file), skiprows=[1], encoding='unicode_escape')
        temp = df['Temp'].values
        rh = df['RH'].values.astype(float)
        p = df['P'].values
        alt = df['HeightMSL'].values
        angle = df['Dir'].values
        speed = df['Speed'].values
        
        #Extend profile to the surface !!! Still creates a linear decrease with height after interpolation!!!
        if minAlt < np.nanmin(alt):
            slp, altN = extendProfile(p, alt, temp, minAlt)
            nans = np.full(len(altN), np.nan)
            temp = np.insert(temp, 0, nans)
            rh = np.insert(rh, 0, nans)
            p = np.insert(p, 0, slp)
            alt = np.insert(alt, 0, altN)
            angle = np.insert(angle, 0, nans)
            speed = np.insert(speed, 0, nans)
            
        simpleProfilePlot([temp, rh, p], alt, ['Temp', 'RH', 'Pressure'], caseID)
        
        return temp, rh, p, alt, speed, angle

    elif file[0:2]=='NI':
        n = 10
        ni = xr.open_dataset(os.path.join(os.getcwd(), str(infodf['Directory'][caseID]), file))
        alt = ni.alt[None:None:n].values
        temp = ni.air_temperature[None:None:n].values
        pres = ni.air_pressure[None:None:n].values
        rh = ni.relative_humidity[None:None:n].values
        temp = temp - 273.15
        speed = ni.wind_speed[None:None:n].values
        angle = ni.wind_direction[None:None:n].values
        #temp, pres, rh = applyQCflag(temp, pres, rh, ni, n)
        
        #Extend profile to the surface (i haven't check to see if this works with NI profile)
        if minAlt < np.nanmin(alt):
            slp, altN = extendProfile(pres, alt, temp, minAlt)
            nans = np.full(len(altN), np.nan)
            temp = np.insert(temp, 0, nans)
            rh = np.insert(rh, 0, nans)
            pres = np.insert(pres, 0, slp)
            alt = np.insert(alt, 0, altN)
            angle = np.insert(angle, 0, nans)
            speed = np.insert(speed, 0, nans)
            
        simpleProfilePlot([temp, rh, pres], alt, ['Temp', 'RH', 'Pressure'], caseID)
        
        return temp, rh, pres, alt, speed, angle
    else:
        print('Unrecognized profile filetype')


# Given a few observed pressures and altitudes, will linearly interpolate pressure
# values to a given grid of altitudes. Theres probably a better mathematical way
# to do this but it will do a calculation based on whether the grid point is above the highest ob (extrapolation),
# below the lowest ob (extrapolation), or between obs (interpolation)
def presInterp(presOb, altOb, altGrid):
    presL = []
    for idx, a in enumerate(altGrid):
        altDifs = [i-a for i in altOb]
        if a <= altOb[0]:
            dA = altOb[1] - altOb[0]
            dP = (presOb[1] - presOb[0]) / dA
            
            da = a - altOb[0]
            pres = presOb[0] + (dP * da)
        elif a >= altOb[-1]:
            dA = altOb[-1] - altOb[-2]
            dP = (presOb[-1] - presOb[-2]) / dA
            
            da = a - altOb[-1]
            pres = presOb[-1] + (dP * da)
        else:
            s = np.where(np.diff(np.sign(altDifs)))[0][0]
            
            dA = (altOb[s+1] - altOb[s])
            dP = (presOb[s+1] - presOb[s]) / dA
            
            da = a - altOb[s]
            pres = presOb[s] + (dP * da)
        presL.append(pres)
        
    return np.asarray(presL)

# This function is wildly overcomplicated but pretty efficient
# This function (which is not very aptly named) gets observations at some distance from the boundary,
# smooths them, performs the RBF, then builds a reference profile.
# 
# If this methodology is ever used again on a larger set of cases, this function should be re-written
# such that: the entire transect is smoothed, RBF interpolated, producing a full objective analysis.
# Then find a window within the objective analysis to use as the reference profile. Rather than producing
# an objective analysis at both the boundary crossing and the reference distance. Trust me, it will simplify
# the code so much and make it better adaptable to more cases.
def getProfileCorr(lf, com, lfUTC, cUTC, minAlt, maxAlt, n, ts, infodf, caseID):
    # The following if/else block just distinguishes the process for 2019 vs 2023 cases
    if caseID in [0, 1, 2, 3, 4, 5, 6, 7, 22]:
        # I fed in a dataframe that contained my target 5km time for each case
        # I found this time manually by looking at time series
        # Grabs the 30 second window around the target time (let's call it the reference window)
        mid = datetime.datetime.strptime(infodf['Profile_Corr_LF1'][caseID].split('-')[0], '%Y%m%d%H%M%S')
        startlf = mid - datetime.timedelta(seconds=15)
        endlf = mid + datetime.timedelta(seconds=15)
        
        mid = datetime.datetime.strptime(infodf['Profile_Corr_Com'][caseID].split('-')[0], '%Y%m%d%H%M%S')
        cstart = mid - datetime.timedelta(seconds=15)
        cend = mid + datetime.timedelta(seconds=15)
        
        # Grab the observations during the reference window, with hysteresis correction applied
        lftau=float(infodf['LF1_TC'][caseID])
        lftemp, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'air_temperature', n, lftau)
        lfalt, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'alt', n, None)
        lfpres, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'air_pressure', n, None)
        lfrh, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'relative_humidity', n, lftau)
        lflat, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'lat', n, None)
        lflon, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'lon', n, None)
        
        lftemp = lftemp - 273.15
        lfThetaV = thetav(lftemp, lfpres, lfrh)
        lfThetaE = thetaE(lftemp, lfpres, lfrh)
        lftemp = lftemp + 273.15
        lf2alt = lfalt
        
        lfu, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'eastward_wind', n, None)
        lfv, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'northward_wind', n, None)
        lfuunc, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'eastward_wind_uncertainty', n, None)
        lfvunc, lfUTC2 = lfPlotables(lf[0], lfUTC[0], startlf, endlf, 'northward_wind_uncertainty', n, None)
        
        calt, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'alt', 1)
        cthetav, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'theta_v', 1)
        cthetae, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'theta_e', 1)
        cpres, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'pressure', 1)
        ctemp, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'fast_temp', 1)
        crh, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'calc_corr_RH', 1)
        cspd, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'wind_speed', 1)
        cdir, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'wind_dir', 1)
        cu, cv = winduv(cspd, cdir)
        clat, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'lat', 1)
        clon, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'lon', 1)
        cpres = cpres / 100
                
        # Find the boundary-relative distance of both platforms to use as the x-grid for the RBF
        lfbdcoord, bidx1 = boundaryRelative(lflat, lflon, lfUTC2, ts, caseID)
        cbdcoord, bidx1 = boundaryRelative(clat, clon, cUTC2, ts, caseID)
        
        # Set the new uniform grid based on the available boundary relative coords
        spacing = 0.025
        newCoords = np.arange(min([np.nanmin(lfbdcoord), np.nanmin(cbdcoord)]), max([np.nanmax(lfbdcoord),np.nanmax(cbdcoord)])+0.025, step=spacing)
        
        # Set the degree of smoothing of observations (essential before RBF use)
        wid = 3
        bdcoord, lfThetaVS = bdSmooth(lfbdcoord, lfThetaV.magnitude, newCoords, wid)
        bdcoord, lfThetaES = bdSmooth(lfbdcoord, lfThetaE.magnitude, newCoords, wid)
        bdcoord, lfpresS = bdSmooth(lfbdcoord, lfpres, newCoords, wid)
        bdcoord, lftempS = bdSmooth(lfbdcoord, lftemp, newCoords, wid)
        bdcoord, lfrhS = bdSmooth(lfbdcoord, lfrh, newCoords, wid)
        bdcoord, lfuS = bdSmoothF(lfbdcoord, lfu, newCoords, wid, lfuunc, 2.0)
        bdcoord, lfvS = bdSmoothF(lfbdcoord, lfv, newCoords, wid, lfvunc, 2.0)
        bdcoord, lfaltS = bdSmooth(lfbdcoord, lfalt, newCoords, wid)
        bdcoord, cthetavS = bdSmooth(cbdcoord, cthetav, newCoords, wid)
        bdcoord, cthetaeS = bdSmooth(cbdcoord, cthetae, newCoords, wid)
        bdcoord, cpresS = bdSmooth(cbdcoord, cpres, newCoords, wid)
        bdcoord, ctempS = bdSmooth(cbdcoord, ctemp, newCoords, wid)
        bdcoord, crhS = bdSmooth(cbdcoord, crh, newCoords, wid)
        bdcoord, cuS = bdSmooth(cbdcoord, cu, newCoords, wid)
        bdcoord, cvS = bdSmooth(cbdcoord, cv, newCoords, wid)
        bdcoord, caltS = bdSmooth(cbdcoord, calt, newCoords, wid)
        
        # Build the x and y grid for the RBF
        x = np.concatenate([np.multiply(bdcoord,1000),
                            np.multiply(bdcoord,1000)])
        y = np.concatenate([caltS, lfaltS])
        
        # An objective analysis and subplot is produced for each dictionary key
        dataDict = {0:[cthetavS, lfThetaVS],
                    1:[cthetaeS, lfThetaES],
                    2:[cpresS, lfpresS],
                    3:[ctempS, lftempS],
                    4:[crhS, lfrhS]}
        
        altDifs = [np.nanmin(calt), np.nanmax(lf2alt)]
        xmin = np.nanmax([np.nanmin(bdcoord[~np.isnan(cpresS)])*1000, np.nanmin(bdcoord[~np.isnan(lfpresS)])*1000])
        xmax = np.nanmin([np.nanmax(bdcoord[~np.isnan(cpresS)])*1000, np.nanmax(bdcoord[~np.isnan(lfpresS)])*1000])
        
        # Extra option parameters used solely for plotting later
        means = [np.asarray([np.nanmean(cthetavS), np.nanmean(lfThetaVS)]),
                  np.asarray([np.nanmean(cthetaeS), np.nanmean(lfThetaES)]),
                  np.asarray([np.nanmean(cpresS), np.nanmean(lfpresS)]),
                  np.asarray([np.nanmean(ctempS), np.nanmean(lftempS)]),
                  np.asarray([np.nanmean(crhS), np.nanmean(lfrhS)]),
                  np.asarray([np.nanmean(cuS), np.nanmean(lfuS)]),
                  np.asarray([np.nanmean(cvS), np.nanmean(lfvS)]),
                  np.asarray([np.nanmean(caltS), np.nanmean(lfaltS)])]
        mins = [np.asarray([np.nanmin(cthetavS), np.nanmin(lfThetaVS)]),
                  np.asarray([np.nanmin(cthetaeS), np.nanmin(lfThetaES)]),
                  np.asarray([np.nanmin(cpresS), np.nanmin(lfpresS)]),
                  np.asarray([np.nanmin(ctempS), np.nanmin(lftempS)]),
                  np.asarray([np.nanmin(crhS), np.nanmin(lfrhS)]),
                  np.asarray([np.nanmin(caltS), np.nanmin(lfaltS)])]
        maxs = [np.asarray([np.nanmax(cthetavS), np.nanmax(lfThetaVS)]),
                  np.asarray([np.nanmax(cthetaeS), np.nanmax(lfThetaES)]),
                  np.asarray([np.nanmax(cpresS), np.nanmax(lfpresS)]),
                  np.asarray([np.nanmax(ctempS), np.nanmax(lftempS)]),
                  np.asarray([np.nanmax(crhS), np.nanmax(lfrhS)]),
                  np.asarray([np.nanmax(caltS), np.nanmax(lfaltS)])]
        
    else: # 2023 cases
        # Process is nearly the same, except now an extra UAS is present
        mid = datetime.datetime.strptime(infodf['Profile_Corr_LF2'][caseID].split('-')[0], '%Y%m%d%H%M%S')
        startlf2 = mid - datetime.timedelta(seconds=15)
        endlf2 = mid + datetime.timedelta(seconds=15)
        
        mid = datetime.datetime.strptime(infodf['Profile_Corr_LF1'][caseID].split('-')[0], '%Y%m%d%H%M%S')
        startlf1 = mid - datetime.timedelta(seconds=15)
        endlf1 = mid + datetime.timedelta(seconds=15)
        
        mid = datetime.datetime.strptime(infodf['Profile_Corr_Com'][caseID].split('-')[0], '%Y%m%d%H%M%S')
        cstart = mid - datetime.timedelta(seconds=15)
        cend = mid + datetime.timedelta(seconds=15)
        
        lf2tau = float(infodf['LF2_TC'][caseID])
        lf2rtau=float(infodf['LF2_RC'][caseID])
        lf2temp, irregUTC = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'air_rh_temp_ipth', n, lf2tau)
        lf2alt, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'alt', n, None)
        lf2pres, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'air_pressure_VN', n, None)
        #lf2pres = lf2pres / 10
        lf2rh, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'air_rh_ipth', n, lf2rtau)
        lf2lat, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'lat', n, None)
        lf2lon, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'lon', n, None)
        lf2u, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'eastward_wind', n, None)
        lf2v, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'northward_wind', n, None)
        lf2uunc, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'eastward_wind_uncertainty', n, None)
        lf2vunc, lf2UTC2 = lfPlotables(lf[0], lfUTC[0], startlf2, endlf2, 'northward_wind_uncertainty', n, None)
        
        # Add RH bias correction
        lf2rh = iprhCorrect(lf2rh, lf2temp)
        
        lf2temp = lf2temp - 273.15

        lf2ThetaV = thetav(lf2temp, lf2pres, lf2rh)
        lf2ThetaE = thetaE(lf2temp, lf2pres, lf2rh)
        lf2temp = lf2temp + 273.15
        
        lf1tau = float(infodf['LF1_TC'][caseID])
        lf1rtau=float(infodf['LF1_RC'][caseID])
        lf1temp, irregUTC = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'air_rh_temp_ipth', n, lf1tau)
        lf1alt, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'alt', n, None)
        lf1pres, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'air_pressure_VN', n, None)
        #lf1pres = lf1pres / 10
        lf1rh, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'air_rh_ipth', n, lf1rtau)
        lf1lat, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'lat', n, None)
        lf1lon, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'lon', n, None)
        lf1u, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'eastward_wind', n, None)
        lf1v, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'northward_wind', n, None)
        lf1uunc, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'eastward_wind_uncertainty', n, None)
        lf1vunc, lf1UTC2 = lfPlotables(lf[1], lfUTC[1], startlf1, endlf1, 'northward_wind_uncertainty', n, None)

        # Add RH bias correction
        lf1rh = iprhCorrect(lf1rh, lf1temp)

        lf1temp = lf1temp - 273.15

        lf1ThetaV = thetav(lf1temp, lf1pres, lf1rh)
        lf1ThetaE = thetaE(lf1temp, lf1pres, lf1rh)
        lf1temp = lf1temp + 273.15
        
        
        calt, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'alt', 1)
        cthetav, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'theta_v', 1)
        cthetae, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'theta_e', 1)
        cpres, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'pressure', 1)
        ctemp, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'fast_temp', 1)
        crh, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'calc_corr_RH', 1)
        cspd, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'wind_speed', 1)
        cdir, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'wind_dir', 1)
        cu, cv = winduv(cspd, cdir)
        clat, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'lat', 1)
        clon, cUTC2 = comPlotables(com, cUTC, cstart, cend, 'lon', 1)
        cpres = cpres / 100
    
        y = np.concatenate([calt, lf1alt, lf2alt])
        
        lfbdcoord2, bidx1 = boundaryRelative(lf2lat, lf2lon, lf2UTC2, ts, caseID)
        lfbdcoord1, bidx1 = boundaryRelative(lf1lat, lf1lon, lf1UTC2, ts, caseID)
        cbdcoord, bidx1 = boundaryRelative(clat, clon, cUTC2, ts, caseID)
        
        spacing = 0.025
        newCoords = np.arange(min([np.nanmin(lfbdcoord2), np.nanmin(lfbdcoord1), np.nanmin(cbdcoord)]), max([np.nanmax(lfbdcoord2), np.nanmax(lfbdcoord1), np.nanmax(cbdcoord)])+spacing, step=spacing)
        wid = 3
        bdcoord, lf2ThetaVS = bdSmooth(lfbdcoord2, lf2ThetaV.magnitude, newCoords, wid)
        bdcoord, lf2ThetaES = bdSmooth(lfbdcoord2, lf2ThetaE.magnitude, newCoords, wid)
        bdcoord, lf2presS = bdSmooth(lfbdcoord2, lf2pres, newCoords, wid)
        bdcoord, lf2tempS = bdSmooth(lfbdcoord2, lf2temp, newCoords, wid)
        bdcoord, lf2rhS = bdSmooth(lfbdcoord2, lf2rh, newCoords, wid)
        bdcoord, lf2uS = bdSmoothF(lfbdcoord2, lf2u, newCoords, wid, lf2uunc, 2.0)
        bdcoord, lf2vS = bdSmoothF(lfbdcoord2, lf2v, newCoords, wid, lf2vunc, 2.0)
        bdcoord, lf2altS = bdSmooth(lfbdcoord2, lf2alt, newCoords, wid)
        bdcoord, lf1ThetaVS = bdSmooth(lfbdcoord1, lf1ThetaV.magnitude, newCoords, wid)
        bdcoord, lf1ThetaES = bdSmooth(lfbdcoord1, lf1ThetaE.magnitude, newCoords, wid)
        bdcoord, lf1presS = bdSmooth(lfbdcoord1, lf1pres, newCoords, wid)
        bdcoord, lf1tempS = bdSmooth(lfbdcoord1, lf1temp, newCoords, wid)
        bdcoord, lf1rhS = bdSmooth(lfbdcoord1, lf1rh, newCoords, wid)
        bdcoord, lf1uS = bdSmoothF(lfbdcoord1, lf1u, newCoords, wid, lf1uunc, 2.0)
        bdcoord, lf1vS = bdSmoothF(lfbdcoord1, lf1v, newCoords, wid, lf1vunc, 2.0)
        bdcoord, lf1altS = bdSmooth(lfbdcoord1, lf1alt, newCoords, wid)
        bdcoord, cthetavS = bdSmooth(cbdcoord, cthetav, newCoords, wid)
        bdcoord, cthetaeS = bdSmooth(cbdcoord, cthetae, newCoords, wid)
        bdcoord, cpresS = bdSmooth(cbdcoord, cpres, newCoords, wid)
        bdcoord, ctempS = bdSmooth(cbdcoord, ctemp, newCoords, wid)
        bdcoord, crhS = bdSmooth(cbdcoord, crh, newCoords, wid)
        bdcoord, cuS = bdSmooth(cbdcoord, cu, newCoords, wid)
        bdcoord, cvS = bdSmooth(cbdcoord, cv, newCoords, wid)
        bdcoord, caltS = bdSmooth(cbdcoord, calt, newCoords, wid)
        
        x = np.concatenate([np.multiply(bdcoord,1000),
                            np.multiply(bdcoord,1000),
                           np.multiply(bdcoord,1000)])
        y = np.concatenate([caltS, lf1altS, lf2altS])
        
        dataDict = {0:[cthetavS, lf1ThetaVS, lf2ThetaVS],
                    1:[cthetaeS, lf1ThetaES, lf2ThetaES],
                    2:[cpresS, lf1presS, lf2presS],
                    3:[ctempS, lf1tempS, lf2tempS],
                    4:[crhS, lf1rhS, lf2rhS]}
        
        altDifs = [np.nanmin(calt), np.nanmax(lf1alt), np.nanmax(lf2alt)]
        xmin = np.nanmax([np.nanmin(bdcoord[~np.isnan(cpresS)])*1000, np.nanmin(bdcoord[~np.isnan(lf1presS)])*1000, np.nanmin(bdcoord[~np.isnan(lf2presS)])*1000])
        xmax = np.nanmin([np.nanmax(bdcoord[~np.isnan(cpresS)])*1000, np.nanmax(bdcoord[~np.isnan(lf1presS)])*1000, np.nanmax(bdcoord[~np.isnan(lf2presS)])*1000])
        print(xmin, xmax)
        
        means = [np.asarray([np.nanmean(cthetavS), np.nanmean(lf1ThetaVS), np.nanmean(lf2ThetaVS)]),
                  np.asarray([np.nanmean(cthetaeS), np.nanmean(lf1ThetaES), np.nanmean(lf2ThetaES)]),
                  np.asarray([np.nanmean(cpresS), np.nanmean(lf1presS), np.nanmean(lf2presS)]),
                  np.asarray([np.nanmean(ctempS), np.nanmean(lf1tempS), np.nanmean(lf2tempS)]),
                  np.asarray([np.nanmean(crhS), np.nanmean(lf1rhS), np.nanmean(lf2rhS)]),
                  np.asarray([np.nanmean(cuS), np.nanmean(lf1uS), np.nanmean(lf2uS)]),
                  np.asarray([np.nanmean(cvS), np.nanmean(lf1vS), np.nanmean(lf2vS)]),
                  np.asarray([np.nanmean(caltS), np.nanmean(lf1altS), np.nanmean(lf2altS)])]
        mins = [np.asarray([np.nanmin(cthetavS), np.nanmin(lf1ThetaVS), np.nanmin(lf2ThetaVS)]),
                  np.asarray([np.nanmin(cthetaeS), np.nanmin(lf1ThetaES), np.nanmin(lf2ThetaES)]),
                  np.asarray([np.nanmin(cpresS), np.nanmin(lf1presS), np.nanmin(lf2presS)]),
                  np.asarray([np.nanmin(ctempS), np.nanmin(lf1tempS), np.nanmin(lf2tempS)]),
                  np.asarray([np.nanmin(crhS), np.nanmin(lf1rhS), np.nanmin(lf2rhS)]),
                  np.asarray([np.nanmin(caltS), np.nanmin(lf1altS), np.nanmin(lf2altS)])]
        maxs = [np.asarray([np.nanmax(cthetavS), np.nanmax(lf1ThetaVS), np.nanmax(lf2ThetaVS)]),
                  np.asarray([np.nanmax(cthetaeS), np.nanmax(lf1ThetaES), np.nanmax(lf2ThetaES)]),
                  np.asarray([np.nanmax(cpresS), np.nanmax(lf1presS), np.nanmax(lf2presS)]),
                  np.asarray([np.nanmax(ctempS), np.nanmax(lf1tempS), np.nanmax(lf2tempS)]),
                  np.asarray([np.nanmax(crhS), np.nanmax(lf1rhS), np.nanmax(lf2rhS)]),
                  np.asarray([np.nanmax(caltS), np.nanmax(lf1altS), np.nanmax(lf2altS)])]
        
    returnList = []
    fig, axes = plt.subplots(len(dataDict.keys()), 1, figsize=(19,13), facecolor ='white', clear=True)
    plt.rcParams.update({'font.size': 20})
    # Will iterate through the variable keys and produce objective analysis plots
    for vidx in range(0, len(dataDict.keys())):
        # Make a meshgrid of the uniform coordinates
        grid_x, grid_y = np.meshgrid(np.arange(xmin, xmax+(spacing*1000), 10), np.arange(minAlt, maxAlt, 1))
        z = np.concatenate(dataDict[vidx])
        
        #Griddata, probably not using this any more
        #grid_z = griddata(np.column_stack((x,y)), z, np.column_stack([grid_x.ravel(), grid_y.ravel()]), fill_value=np.nan,method='linear')
        
        # Remove nan before RBFing
        xF1, yF1, zF1 = remove_nan_observations(x, y, z)
        xF, zF, yF = remove_nan_observations(xF1, zF1, yF1)
        
        # Scale your values to 0-1 before inputting to the RBF.
        # Honestly don't think this does much so long as the inputs are pre-smoothed (in my case they are)
        zscale = (zF - min(mins[vidx])) / (max(maxs[vidx]) - min(mins[vidx]))
        rbf = Rbf(xF, yF, zscale, function='thin_plate')
        grid_z = rbf(grid_x, grid_y)
        # Restore scaled values back to original
        grid_z = grid_z.ravel() * (max(maxs[vidx]) - min(mins[vidx])) + min(mins[vidx])
        
        #Barnes, again probably not using this any more
        #kappaAlt = np.nanmax(y) - np.nanmin(y)
        #kappaAlt = maxAltDiff(altDifs)
        #grid_z = barnesA(x, y, z, grid_x, grid_y, kappaAlt, 0, 2, 0.5, plot=True)
        
        # Remove nans again for some reason
        xx, yy, zz = remove_nan_observations(grid_x.ravel(), grid_y.ravel(), grid_z)
        
        # This part builds the reference profile. At this stage you have an objective analysis surface,
        # but what you want is a profile with some temp, theta, etc. for each meter altitude
        zavA = np.asarray([])
        zaltA = np.asarray([])
        counter = 0
        for yidx, yrow in enumerate(grid_y):
            zaltA = np.append(zaltA, yrow[0])
            xrow = grid_x[yidx]
            zr = grid_z[counter:counter+len(xrow)+1]
            # Append the mean value at that height on the surface
            zavA = np.append(zavA, np.nanmean(zr))
            
            # Add the averages as a colored bar on the side of the plot
            axes[vidx].scatter(x=xmax+15, y=yrow[0], c=np.nanmean(zr), vmin=np.nanmin(grid_z), vmax=np.nanmax(grid_z))
            counter+=len(xrow)
            
        # Plot the objective analysis surface
        f = axes[vidx].tricontourf(xx, yy, zz, levels=25)
        axes[vidx].scatter(x, y, c=z, s=100, zorder=4, vmin=np.nanmin(grid_z), vmax=np.nanmax(grid_z))
        axes[vidx].scatter(x, y, s=200, c='black', zorder=3, alpha=0.3)
        divider = make_axes_locatable(axes[vidx])
        cax = divider.append_axes("right", size="3%", pad=0.1)
        cbar = plt.colorbar(f, cax=cax)
        
        returnList.append(zavA)
    
    # Pressure is different. Don't use the RBF, linearity needs to be forced.        
    presInt = presInterp(means[2], means[-1], zaltA)
        
    axes[-1].set_xlabel('Boundary Relative Distance')
    fig.tight_layout()
    
    # Return everything in the weird numpy unflattened shape I for some reason built into deficit()
    return returnList[0].reshape(len(zaltA), 1), returnList[1].reshape(len(zaltA), 1), presInt.reshape(len(zaltA), 1), returnList[3].reshape(len(zaltA), 1), returnList[4].reshape(len(zaltA), 1), zaltA.reshape(len(zaltA), 1), means

# Don't remember what this does, probably not important
def applyQCflag(temp, pres, rh, ds, n):
    tflag = ds.air_temperature_flag[None:None:n].values
    tempc = temp
    for idx, t in enumerate(tempc):
        if tflag[idx]!=0:
            temp[idx] = np.nan

    rflag = ds.relative_humidity_flag[None:None:n].values
    rhc = rh
    for idx, t in enumerate(tempc):
        if rflag[idx]!=0:
            rh[idx] = np.nan
            
    pflag = ds.air_pressure_flag[None:None:n].values
    presc = pres
    for idx, t in enumerate(presc):
        if pflag[idx]!=0:
            pres[idx] = np.nan
            
    return temp, pres, rh

# Returns a dictionary of datetime objects according to the boundary crit-times I have in infodf
def critTimes(df, caseID):
    fullString = df['Crit_Times'][caseID]
    transectSets = fullString.split(':')
    boundaryDict = {}
    for idx, transect in enumerate(transectSets):
        timeStrings = transect.split('-')
        boundary = []
        for timeString in timeStrings:
            dt = datetime.datetime.strptime(timeString, '%Y%m%d%H%M')
            boundary.append(dt)
        boundaryDict[idx] = boundary
        
    return boundaryDict

# This function take the meso location "keyframes" and interpolates and/or extrapolates
# to any given new observation time    
def newMesoCoords(curtime, caseID):
    mlat, mlon, mtime = mCoords(caseID)
    
    timeDifs = [(i-curtime).total_seconds() for i in mtime]
    
    if curtime <= mtime[0]:
        dt = (mtime[1] - mtime[0]).total_seconds()
        dlat = (mlat[1] - mlat[0]) / dt
        dlon = (mlon[1] - mlon[0]) / dt
        
        ds = (curtime - mtime[0]).total_seconds()
        mlat = mlat[0] + (dlat * ds)
        mlon = mlon[0] + (dlon * ds)
    elif curtime >= mtime[-1]:
        dt = (mtime[-1] - mtime[-2]).total_seconds()
        dlat = (mlat[-1] - mlat[-2]) / dt
        dlon = (mlon[-1] - mlon[-2]) / dt
        
        ds = (curtime - mtime[-1]).total_seconds()
        mlat = mlat[-1] + (dlat * ds)
        mlon = mlon[-1] + (dlon * ds)
    else:
        s = np.where(np.diff(np.sign(timeDifs)))[0][0]
        
        dt = (mtime[s+1] - mtime[s]).total_seconds()
        dlat = (mlat[s+1] - mlat[s]) / dt
        dlon = (mlon[s+1] - mlon[s]) / dt
        
        ds = (curtime - mtime[s]).total_seconds()
        mlat = mlat[s] + (dlat * ds)
        mlon = mlon[s] + (dlon * ds)
    
    return mlat, mlon

# Finds the storm motion based on a few moeso locations and times
def stormMotion(curtime, caseID):
    mlat, mlon, mtime = mCoords(caseID)
    uList = []
    vList = []
    
    # This equates longitude lines to be aligned with UAS wind 0 degrees (bad??)
    for idx, time in enumerate(mtime[0:-1]):
        udist = getDistance(mlat[idx], mlon[idx], mlat[idx], mlon[idx+1])
        vdist = getDistance(mlat[idx], mlon[idx], mlat[idx+1], mlon[idx])
        if mlat[idx] > mlat[idx+1]:
            vdist = vdist * -1
        if mlon[idx] > mlon[idx+1]:
            udist = udist * -1
        ds = (mtime[idx+1] - time).total_seconds()
        u = (udist * 1000) / ds
        v = (vdist * 1000) / ds
        uList.append(u)
        vList.append(v)
        
    timeDifs = [(i-curtime).total_seconds() for i in mtime]
    if curtime <= mtime[0]:
        return uList[0], vList[0]
    elif curtime >= mtime[-1]:
        return uList[-1], vList[-1]
    else:
        s = np.where(np.diff(np.sign(timeDifs)))[0][0]
        return uList[s], vList[s]

# Not sure how this is different from the previous function. Might be the averaged storm motion
def stormMotionDur(caseID):
    mlat, mlon, mtime = mCoords(caseID)
    uList = []
    vList = []
    
    # This equates longitude lines to be aligned with UAS wind 0 degrees
    udist = getDistance(mlat[0], mlon[0], mlat[0], mlon[1])
    vdist = getDistance(mlat[0], mlon[0], mlat[1], mlon[0])
    if mlat[0] > mlat[1]:
        vdist = vdist * -1
    if mlon[0] > mlon[1]:
        udist = udist * -1
    ds = (mtime[1] - mtime[0]).total_seconds()
    u = (udist * 1000) / ds
    v = (vdist * 1000) / ds
    
    print(f'Storm motion is {u*1.94} knots east, {v*1.94} knots north')
        
    return u, v

# Coordinate "keyframes" are stored in boundaryCoords.py
# This function takes each of the 4 coordinates along the boundary and interpolates and/or extrapolates
# to any given new observation time
def newBdCoords(curtime, caseID):
    bdlat, bdlon, bdtime = bdCoords(caseID)
    
    timeDifs = [(i-curtime).total_seconds() for i in bdtime]
    
    if curtime <= bdtime[0]:
        dt = (bdtime[1] - bdtime[0]).total_seconds()
        dlat = (bdlat[1] - bdlat[0]) / dt
        dlon = (bdlon[1] - bdlon[0]) / dt
        
        ds = (curtime - bdtime[0]).total_seconds()
        bdlat = bdlat[0] + (dlat * ds)
        bdlon = bdlon[0] + (dlon * ds)
    elif curtime >= bdtime[-1]:
        dt = (bdtime[-1] - bdtime[-2]).total_seconds()
        dlat = (bdlat[-1] - bdlat[-2]) / dt
        dlon = (bdlon[-1] - bdlon[-2]) / dt
        
        ds = (curtime - bdtime[-1]).total_seconds()
        bdlat = bdlat[-1] + (dlat * ds)
        bdlon = bdlon[-1] + (dlon * ds)
    else:
        s = np.where(np.diff(np.sign(timeDifs)))[0][0]
        
        dt = (bdtime[s+1] - bdtime[s]).total_seconds()
        dlat = (bdlat[s+1] - bdlat[s]) / dt
        dlon = (bdlon[s+1] - bdlon[s]) / dt
        
        ds = (curtime - bdtime[s]).total_seconds()
        bdlat = bdlat[s] + (dlat * ds)
        bdlon = bdlon[s] + (dlon * ds)
    
    return bdlat, bdlon

# Take the low-resolution 4-points along the boundary and increases it to 40 to ensure
# when the "nearest point" on the boundary is found, it is actually the nearest point
def highBdCoords(bdlat, bdlon):
    latdif = []
    londif = []
    res = 40
    for idx, i in enumerate(bdlat[0:len(bdlat)-1]):
        latdif.append((i - bdlat[idx+1]) / res)
    for idx, i in enumerate(bdlon[0:len(bdlon)-1]):
        londif.append((i - bdlon[idx+1]) / res)
    
    newLats = []
    newLons = []
    for idx, dif in enumerate(latdif):
        for n in range(0, res):
            newLats.append(bdlat[idx]-(latdif[idx]*n))
            newLons.append(bdlon[idx]-(londif[idx]*n))
            
    return np.asarray(newLats), np.asarray(newLons)

# Given the high resolution 40 points on the boundary, find the nearest point to an array of
# observation lats, lons, times and calculate the distance.
def boundaryRelative(ilats, ilons, iUTC, ts, caseID):
    bdRelCoord = []
    bidx = []
    # torient looks in infodf and finds whether the boundary is oriented more north-south or east-west
    orient = torient(ts, caseID)
    print('Solving boundary relative coordinates')
    for idx, lat in enumerate(ilats):
        curtime = iUTC[idx]
        bdlat, bdlon = newBdCoords(curtime, caseID)
        bdlatH, bdlonH = highBdCoords(bdlat, bdlon)
        distList = []
        for hidx, hlat in enumerate(bdlatH):
            dist = getDistance(lat, ilons[idx], hlat, bdlonH[hidx])
            distList.append(dist.data.tolist())
        
        # Block controls when the boundary has been passed based on transect orientation
        if orient:
            if ilats[idx] > bdlatH[distList.index(min(distList))]:
                bdRelCoord.append(min(distList)*-1)
            else:
                bdRelCoord.append(min(distList))
        else:
            if ilons[idx] < bdlonH[distList.index(min(distList))]:
                bdRelCoord.append(min(distList)*-1)
            else:
                bdRelCoord.append(min(distList))
                
        # Return index of the point used
        bidx.append(distList.index(min(distList)))

    return bdRelCoord, bidx

# solves for the i,j unit vector boundary normal orientation using the nearest point on the boundary
def borient(ilats, ilons, iUTC, caseID):
    # Boundary orientation unit vector list (90 degrees normal)
    print('Solving boundary orientations')
    vectList = []
    for idx, lat in enumerate(ilats):
        curtime = iUTC[idx]
        bdlat, bdlon = newBdCoords(curtime, caseID)
        bdlatH, bdlonH = highBdCoords(bdlat, bdlon)
        distList = []
        for hidx, hlat in enumerate(bdlatH):
            dist = getDistance(lat, ilons[idx], hlat, bdlonH[hidx])
            distList.append(dist.data.tolist())
        
        minidx = distList.index(min(distList))
        
        # Extra check for cases where nearest point is boundary edge
        if minidx == 0:
            minidx+=1
        elif minidx == len(distList)-1:
            minidx-=1
        
        # From minidx-1 to minidx+1
        udist = getDistance(bdlatH[minidx-1], bdlonH[minidx-1], bdlatH[minidx-1], bdlonH[minidx+1])
        vdist = getDistance(bdlatH[minidx-1], bdlonH[minidx-1], bdlatH[minidx+1], bdlonH[minidx-1])
        tdist = getDistance(bdlatH[minidx-1], bdlonH[minidx-1], bdlatH[minidx+1], bdlonH[minidx+1])
        
        # Ensure negatives are correct
        if bdlatH[minidx-1] > bdlatH[minidx+1]:
            vdist = vdist * -1
        
        # Convert to unit vector and rotate 90 degrees right
        normx = (1 * vdist) / tdist
        normy = (-1 * udist) / tdist
        vectList.append((normx, normy))
    return vectList

def torient(ts, caseID):
    # Transect orientation boolean
    infodf = pd.read_csv(os.path.join(os.getcwd(), 'CaseInfo.csv'))
    orient = infodf['Torient'][caseID].split('-')[ts]
    if int(orient)==0:
        print('dlat transect')
        return True
    else:
        print('dlon transect')
        return False

# Calculate u/v components from speed, direction
def winduv(inspd, indir):
    u,v = wind_components(inspd * units('m/s'), indir * units.deg)
    return u.magnitude, v.magnitude

# Calculate speed, direction from u/v components
def windA(inu, inv):
    angle = wind_direction(inu*units('m/s'), inv*units('m/s'))
    speed = wind_speed(inu*units('m/s'), inv*units('m/s'))
    return speed.magnitude, angle.magnitude

# Derive horizontal vorticity. This function is for the case when one RAAVEN is present and W gradients
# are just based on one UAS
def hvort2(upperU, upperV, upperW, upperAlt, upperCoord, lowerU, lowerV, lowerAlt, lowerCoord, bdVector, caseID, uunc, vunc, wunc, stat):
    # Using vertical motion assuming changes in w are in the boundary normal direction
    cbd = list(lowerCoord)
    # Option to account for storm motion in streamwise vorticity derivation
    if stat:
        su = 0
        sv = 0
    else:
        su, sv = stormMotionDur(caseID)
    hvort = []
    swvort = []
    swVertL = []
    swShearL = []
    swVertPerc = []
    swShearPerc = []
    swPerc = []
    
    cwvort = []
    cwVertL = []
    cwShearL = []
    cwVertPerc = []
    cwShearPerc = []
    cwPerc = []
    
    swuncL = []
    wxL = []
    wyL = []
    
    wnL = []

    meterCoord = np.multiply(upperCoord, 1000)
    for idx, uAlt in enumerate(upperAlt):
        # Need a nan value at start/end because center differencing
        if idx == 0 or idx == len(upperAlt)-1:
            hvort.append(np.nan)
            swVertL.append(np.nan)
            swShearL.append(np.nan)
            swvort.append(np.nan)
            swVertPerc.append(np.nan)
            swShearPerc.append(np.nan)
            swPerc.append(np.nan)
            
            cwVertL.append(np.nan)
            cwShearL.append(np.nan)
            cwvort.append(np.nan)
            cwVertPerc.append(np.nan)
            cwShearPerc.append(np.nan)
            cwPerc.append(np.nan)
            
            swuncL.append(np.nan)
            wxL.append(np.nan)
            wyL.append(np.nan)
            wnL.append(np.nan)
        
        else:
            coordIdx = lowerCoord.index(upperCoord[idx])
            wx =((upperW[idx+1] - upperW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][1]) - ((upperV[idx] - lowerV[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
            wy = ((upperU[idx] - lowerU[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx])) - ((upperW[idx+1] - upperW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][0])
            
            mag = math.sqrt((wx**2)+(wy**2))
            hvort.append(mag)
            wxL.append(wx)
            wyL.append(wy)
            
            # Possible that there is an extra negative sign in this sector, I flipped bdvector by 180 degrees
            vertxTerm = ((upperW[idx+1] - upperW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][1])
            vertyTerm = -1 * ((upperW[idx+1] - upperW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][0])
            shearxTerm = -1 * ((upperV[idx] - lowerV[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
            shearyTerm = ((upperU[idx] - lowerU[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
    
            # Derive streamwise horizontal vorticity
            midptU = (lowerU[coordIdx] + upperU[idx]) / 2
            midptV = (lowerV[coordIdx] + upperV[idx]) / 2
            udif = midptU - su
            vdif = midptV - sv
            absdif = math.sqrt((udif**2)+(vdif**2))
            sw = ((udif * wx) + (vdif * wy)) / absdif
            swvort.append(sw)
            
            # Derive SW contributions
            swVert = ((udif * vertxTerm) + (vdif * vertyTerm)) / absdif
            swShear = ((udif * shearxTerm) + (vdif * shearyTerm)) / absdif
            swVertPerc.append((swVert / sw))
            swShearPerc.append((swShear / sw))
            swPerc.append((sw / mag))
            swVertL.append(swVert)
            swShearL.append(swShear)
            
            # Derive crosswise horizontal vorticity, 90 degree counter-clockwise rotation from mean wind
            udif90 = -vdif
            vdif90 = udif
            cw = ((udif90 * wx) + (vdif90 * wy)) / absdif
            cwvort.append(cw)
            
            # Derive CW contributions
            cwVert = ((udif90 * vertxTerm) + (vdif90 * vertyTerm)) / absdif
            cwShear = ((udif90 * shearxTerm) + (vdif90 * shearyTerm)) / absdif
            cwVertPerc.append((cwVert / cw))
            cwShearPerc.append((cwShear / cw))
            cwPerc.append((cw / mag))
            cwVertL.append(cwVert)
            cwShearL.append(cwShear)
            
            # Derive boundary-normal vorticity
            wn = ((bdVector[idx][0] * wx) + (bdVector[idx][1] * wy)) / math.sqrt((bdVector[idx][0]**2)+(bdVector[idx][1]**2))
            wnL.append(wn)
            
            # Uncertainty propagation (barf emoji)
            wxunc = np.linalg.norm((np.linalg.norm((wunc[idx+1], wunc[idx-1])) / abs((-1 * (meterCoord[idx+1] - meterCoord[idx-1])) * bdVector[idx][1]),
                                   vunc[idx] / abs((upperAlt[idx] - lowerAlt[coordIdx]))))
            wyunc = np.linalg.norm((uunc[idx] / abs(upperAlt[idx] - lowerAlt[coordIdx]),
                                    np.linalg.norm((wunc[idx+1], wunc[idx-1])) / abs((-1 * (meterCoord[idx+1] - meterCoord[idx-1])) * bdVector[idx][0])))
            swunc = np.linalg.norm((abs(udif) * wxunc, abs(vdif) * wyunc)) / absdif
            swuncL.append(swunc)
            
    # Returns:
    # (L is used throughout to indicate this is the lower comet-UAS part of the profile)
    # hvort: Total horizontal vorticity magnitude
    # swvort, swPerc: Streamwise vorticity magnitude, percent streamwise
    # swVertL, swShearL: magnitude contribution to the streamwise vorticity by vertical motion gradients, vertical shear
    # swVertPerc, swShearPerc: percent contribution to the streamwise vorticity by vertical motion gradients, vertical shear
    # cwvort, cwVertL, cwShearL, cwPerc, cwVertPerc, cwShearPerc: Same as above but for crosswise vorticity
    # swuncL: Uncertainty of streamwise vorticity (not correct, don't use)
    # wxL, wyL, wnL: Horizontal vorticity x component, y, component, and boundary-normal component
    return hvort, swvort, swVertL, swShearL, swPerc, swVertPerc, swShearPerc, cwvort, cwVertL, cwShearL, cwPerc, cwVertPerc, cwShearPerc, swuncL, wxL, wyL, wnL

# Same as hvort2 but accounting for having two vertical motion observation levels
def hvortR(upperU, upperV, upperW, upperAlt, upperCoord, lowerU, lowerV, lowerW, lowerAlt, lowerCoord, bdVector, caseID, stat):
    # Using vertical motion assuming changes in w are in the boundary normal direction
    cbd = list(lowerCoord)
    if stat:
        su = 0
        sv = 0
    else:
        su, sv = stormMotionDur(caseID)
    hvort = []
    swvort = []
    swVertL = []
    swShearL = []
    swVertPerc = []
    swShearPerc = []
    swPerc = []
    
    cwvort = []
    cwVertL = []
    cwShearL = []
    cwVertPerc = []
    cwShearPerc = []
    cwPerc = []
    
    wxL = []
    wyL = []
    
    meterCoord = np.multiply(upperCoord, 1000)
    # Horizontal vort between two RAAVENs allows averaging of w for midpoint instead of making an assumption about midpoint w
    meanW  = (np.asarray(upperW) + np.asarray(lowerW)) / 2
    
    for idx, uAlt in enumerate(upperAlt):
        if idx == 0 or idx == len(upperAlt)-1:
            hvort.append(np.nan)
            swVertL.append(np.nan)
            swShearL.append(np.nan)
            swvort.append(np.nan)
            swVertPerc.append(np.nan)
            swShearPerc.append(np.nan)
            swPerc.append(np.nan)
            
            cwVertL.append(np.nan)
            cwShearL.append(np.nan)
            cwvort.append(np.nan)
            cwVertPerc.append(np.nan)
            cwShearPerc.append(np.nan)
            cwPerc.append(np.nan)
            
            wxL.append(np.nan)
            wyL.append(np.nan)
        else:
            coordIdx = lowerCoord.index(upperCoord[idx])
            # nan is produced when distance is negative for dlon ts1
            wx =((meanW[idx+1] - meanW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][1]) - ((upperV[idx] - lowerV[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
            wxvert = ((meanW[idx+1] - meanW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][1])
            wxwind = - ((upperV[idx] - lowerV[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
            wy = ((upperU[idx] - lowerU[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx])) - ((meanW[idx+1] - meanW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][0])
            mag = math.sqrt((wx**2)+(wy**2))
            hvort.append(mag)
            
            wxL.append(wx)
            wyL.append(wy)
            
            # Uncertainty propagation (barf emoji)
            '''
            wxunc = np.linalg.norm((np.linalg.norm((wunc[idx+1], wunc[idx-1])) / abs((-1 * (meterCoord[idx+1] - meterCoord[idx-1])) * bdVector[idx][1]),
                                   np.linalg.norm((vunc[idx], vunc[coordIdx])) / abs((upperAlt[idx] - lowerAlt[coordIdx]))))
            wyunc = np.linalg.norm((uunc[idx] / abs(upperAlt[idx] - lowerAlt[coordIdx]),
                                    np.linalg.norm((wunc[idx+1], wunc[idx-1])) / abs((-1 * (meterCoord[idx+1] - meterCoord[idx-1])) * bdVector[idx][0])))
            swunc = np.linalg.norm((abs(udif) * wxunc, abs(vdif) * wyunc)) / absdif
            swuncL.append(swunc)
            '''
            
            vertxTerm = ((meanW[idx+1] - meanW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][1])
            vertyTerm = -1 * ((meanW[idx+1] - meanW[idx-1]) / (meterCoord[idx+1] - meterCoord[idx-1]) * bdVector[idx][0])
            
            shearxTerm = -1 * ((upperV[idx] - lowerV[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
            shearyTerm = ((upperU[idx] - lowerU[coordIdx]) / (upperAlt[idx] - lowerAlt[coordIdx]))
    
            # Derive streamwise horizontal vorticity
            midptU = (lowerU[coordIdx] + upperU[idx]) / 2
            midptV = (lowerV[coordIdx] + upperV[idx]) / 2
            udif = midptU - su
            vdif = midptV - sv
            absdif = math.sqrt((udif**2)+(vdif**2))
            sw = ((udif * wx) + (vdif * wy)) / absdif
            swvort.append(sw)
            
            # Derive contributions
            swVert = ((udif * vertxTerm) + (vdif * vertyTerm)) / absdif
            swShear = ((udif * shearxTerm) + (vdif * shearyTerm)) / absdif
            swVertPerc.append((swVert / sw))
            swShearPerc.append((swShear / sw))
            swPerc.append((sw / mag))
            swVertL.append(swVert)
            swShearL.append(swShear)
            
            # Derive crosswise horizontal vorticity, 90 degree counter-clockwise rotation from mean wind
            udif90 = -vdif
            vdif90 = udif
            cw = ((udif90 * wx) + (vdif90 * wy)) / absdif
            cwvort.append(cw)
            
            # Derive CW contributions
            cwVert = ((udif90 * vertxTerm) + (vdif90 * vertyTerm)) / absdif
            cwShear = ((udif90 * shearxTerm) + (vdif90 * shearyTerm)) / absdif
            cwVertPerc.append((cwVert / cw))
            cwShearPerc.append((cwShear / cw))
            cwPerc.append((cw / mag))
            cwVertL.append(cwVert)
            cwShearL.append(cwShear)
        
    return hvort, swvort, swVertL, swShearL, swPerc, swVertPerc, swShearPerc, cwvort, cwVertL, cwShearL, cwPerc, cwVertPerc, cwShearPerc, wxL, wyL

# Derive boundary relative vertical vorticity
def vvort(u, v, coords, bdVector, uunc, vunc):
    vortList = []
    p1Lu = []
    p1Lv = []
    p2Lu = []
    p2Lv = []
    vortuncL = []
    meterCoord = np.multiply(coords, 1000)
    for idx, coord in enumerate(coords):
        if idx == 0 or idx == len(coords)-1:
            vortList.append(np.nan)
            vortuncL.append(np.nan)
        else:
            # bdVector is the normal vector, it is rotated back to parallel here
            bdU = bdVector[idx][1] * -1
            bdV = bdVector[idx][0]
            denom = (bdU**2) + (bdV**2)
            
            # Project uv wind vector onto boundary unit vector
            p1 = np.dot((u[idx-1], v[idx-1]), (bdU, bdV))
            p2 = np.dot((u[idx+1], v[idx+1]), (bdU, bdV))        
            dPmag = p2-p1
            
            vort = dPmag / (meterCoord[idx+1] - meterCoord[idx-1])
            vortList.append(vort)
    
    # Optional plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,11), facecolor ='white', clear=True)
    plt.suptitle(f'Vorticity bug zapper')
    
    ax1.set_ylabel('Wind (m/s)')
    ax1.scatter(x=meterCoord, y=u, c='red', zorder=4)
    ax1.scatter(x=meterCoord, y=v, c='orange', zorder=4)
    s = 10
    ax1.legend(['U', 'V'])
    ax1.set_ylim(ax1.get_ylim())
    ax1.invert_xaxis()
    #ax1.set_xlim([0, -1200])
    ax1.axhline(0)
    
    ax2.scatter(x=meterCoord, y=vortList, c='purple', zorder=5)
    ax2.barbs(meterCoord[None:None:s], vortList[None:None:s], u[None:None:s], v[None:None:s], length=9, zorder=5)
    ax2.set_ylabel('Vertical Vorticity (s-1)')
    ax2.set_xlabel('Distance from Boundary (m)')
    ax2.invert_xaxis()
    ax2.set_xlim(ax1.get_xlim())
    ax2.set_ylim([-0.06, 0.06])
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend(['Vorticity', 'Barbs']).set_zorder(9)
    
    plt.rcParams.update({'font.size': 20})
    #plt.savefig(os.path.join(os.getcwd(),'Figures',str(infodf['Directory'][caseID]), f'{lfUTC2[0]:%Y%m%d-%H%M}_TS{ts}_SWHXVorticityTerms.png'), bbox_inches='tight')
    plt.show()
            
    return vortList

# Derive boundary-relative divergence
def divergence(u, v, coords, bdVector):
    divList = []
    p1magL = []
    p2magL = []
    p1Lu = []
    p1Lv = []
    p2Lu = []
    p2Lv = []
    meterCoord = np.multiply(coords, 1000)
    for idx, coord in enumerate(coords):
        if idx == 0 or idx == len(coords)-1:
            divList.append(np.nan)
        else:
            # bdVector is the normal vector, use normal vector for divergence
            bdU = bdVector[idx][0]
            bdV = bdVector[idx][1]
            denom = (bdU**2) + (bdV**2)
            
            # Project uv wind vector onto boundary unit vector
            p1 = np.dot((u[idx-1], v[idx-1]), (bdU, bdV))
            p2 = np.dot((u[idx+1], v[idx+1]), (bdU, bdV))        
            dPmag = p2-p1
            
            div = dPmag / (meterCoord[idx+1] - meterCoord[idx-1])
            divList.append(div)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,11), facecolor ='white', clear=True)
    plt.suptitle(f'Divergence bug zapper')
    
    ax1.set_ylabel('Wind (m/s)')
    ax1.scatter(x=meterCoord, y=u, c='red', zorder=4)
    ax1.scatter(x=meterCoord, y=v, c='orange', zorder=4)
    s = 7
    #ax1.barbs(meterCoord[1:-1][None:None:s], v[1:-1][None:None:s], np.multiply(p1Lu[None:None:s], 10), np.multiply(p1Lv[None:None:s], 10), length=9, zorder=5)
    #ax1.barbs(meterCoord[1:-1][None:None:s], u[1:-1][None:None:s], np.multiply(p2Lu[None:None:s], 10), np.multiply(p2Lv[None:None:s], 10), length=9, zorder=5)
    ax1.legend(['U', 'V'])
    ax1.set_ylim(ax1.get_ylim())
    ax1.invert_xaxis()
    #ax1.set_xlim([0, -1200])
    #ax1.set_ylim([-8, 6])
    ax1.axhline(0)
    
    ax2.scatter(x=meterCoord, y=divList, c='purple', zorder=5)
    ax2.barbs(meterCoord[None:None:s], divList[None:None:s], u[None:None:s], v[None:None:s], length=9, zorder=5)
    ax2.set_ylabel('Divergence (s-1)')
    ax2.set_xlabel('Distance from Boundary (m)')
    ax2.invert_xaxis()
    ax2.set_xlim(ax1.get_xlim())
    #ax2.set_ylim([-0.06, 0.06])
    ax2.set_xlim(ax1.get_xlim())
    ax2.legend(['Divergence', 'Barbs']).set_zorder(9)
    
    plt.rcParams.update({'font.size': 20})
    #plt.savefig(os.path.join(os.getcwd(),'Figures',str(infodf['Directory'][caseID]), f'{lfUTC2[0]:%Y%m%d-%H%M}_TS{ts}_SWHXVorticityTerms.png'), bbox_inches='tight')
    plt.show()
            
    return divList


# Same as bdSmooth(), but with an option to filter out values based on some 2nd array
# Filtered values that are above some threshold won't be weighted towards the smoothed value
# My main use for this was filtering out high uncertainty wind values based on some threshold
def bdSmoothF(coords, var, newCoords, wid, unc, thresh):
    binSize = 0.025
    newVar = []
    gausAv = []
    maskAv = []

    mu = 0
    sigma = (wid*binSize) / 6

    boxcarWidth = binSize * wid
    for jidx, i in enumerate(newCoords):
        boxcarStart = i - boxcarWidth
        boxcarEnd = i + boxcarWidth
        binVar = [var[idx] for idx, d in enumerate(coords) if boxcarStart <= d <= boxcarEnd and abs(unc[idx])<thresh]
        binDist = [i-d for idx, d in enumerate(coords) if boxcarStart <= d <= boxcarEnd and abs(unc[idx])<thresh]
        binDist = np.asarray(binDist)
        
        mu = 0
        weights = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((binDist - mu) / sigma) ** 2)

        weightAvg = np.nansum(binVar*weights) / np.nansum(weights[~np.isnan(binVar)])
        
        if not list(binDist[~np.isnan(binVar)]):
            weightAvg = np.nan
        elif np.nanmin(abs(binDist[~np.isnan(binVar)]))>binSize:
            weightAvg = np.nan
    
        gausAv.append(weightAvg)
        
    fig, (ax1) = plt.subplots(1, 1, figsize=(12,11), facecolor ='white', clear=True)
    plt.suptitle(f'Boundary Relative Smoothing')
    ax1.set_ylabel('Vertical Wind (m/s)')
    ax1.scatter(x=np.asarray(coords), y=var, c='red', zorder=3)
    ax1.scatter(x=newCoords, y=gausAv, c='green', zorder=5)
    ax1.legend(['OG','Gaussian Average'])
    ax1.set_xlabel('Boundary Relative Distance (km)')
    ax1.set_ylim(ax1.get_ylim())
    #ax1.set_ylim([-1, 2.5])
    ax1.invert_xaxis()
    plt.show()

    return newCoords, gausAv

# Very important function for my thesis. Will take raw observations and smooth
# and regrid to a new coordinate system using gaussian-weighted bin-based averaging.
# Inputs:
#    coords: array of coordinates using some cartesian system (i.e. boundary-relative)
#    var: array of observations at each coordinate
#    newCoords: some new grid to smooth to
#    wid: width of the gaussian curve (larger numbers increases smoothing)
def bdSmooth(coords, var, newCoords, wid):
    # Make sure your bin size matches the grid spacing, in this case 25m
    binSize = 0.025
    newVar = []
    gausAv = []
    
    mu = 0
    # sigma defines the width of the gaussian curve
    sigma = (wid*binSize) / 6
    
    boxcarWidth = binSize * wid
    # Iterates over the uniform coordinate grid where point i is the center of
    # each gaussian curve of weights
    for jidx, i in enumerate(newCoords):
        boxcarStart = i - boxcarWidth
        boxcarEnd = i + boxcarWidth
        # binVar is the observations within the boxcar
        binVar = [var[idx] for idx, d in enumerate(coords) if boxcarStart <= d <= boxcarEnd]
        binDist = [i-d for idx, d in enumerate(coords) if boxcarStart <= d <= boxcarEnd]
        binDist = np.asarray(binDist)
        
        mu = 0
        # Distance from the center i is used as the weight according to the curve
        weights = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((binDist - mu) / sigma) ** 2)

        # Sum the variable*weights and divide by the total weight for the
        # smoothed value
        weightAvg = np.nansum(binVar*weights) / np.nansum(weights[~np.isnan(binVar)])
        
        # Unsure if this first if ever does anything
        if not list(binDist[~np.isnan(binVar)]):
            weightAvg = np.nan
        # Elif will force a nan value at that point if the smallest non-nan
        # distance is beyond the binSize
        elif np.nanmin(abs(binDist[~np.isnan(binVar)]))>binSize:
            weightAvg = np.nan
    
        gausAv.append(weightAvg)
        
    # Plot
    fig, (ax1) = plt.subplots(1, 1, figsize=(12,11), facecolor ='white', clear=True)
    plt.suptitle(f'Boundary Relative Smoothing')
    ax1.set_ylabel('Variable')
    ax1.scatter(x=np.asarray(coords), y=var, c='red', zorder=3)
    ax1.scatter(x=newCoords, y=gausAv, c='blue', zorder=5)
    ax1.legend(['OG','Gaussian Average'])
    ax1.set_xlabel('Boundary Relative Distance (km)')
    ax1.set_ylim(ax1.get_ylim())
    ax1.invert_xaxis()
    plt.show()

    return newCoords, gausAv

# Built for smoothing raw observations in
# preparation for a hysteresis correction in the time dimension.
# Inputs:
#   liUTC: list of datetime objects
#   var: Observations numpy array
#   boxcarWidth: tau or time constant
#   binSize: sampling rate in seconds
def gaussianTime(liUTC, var, boxcarWidth, binSize):
    
    # Make a list that serves as the x coord for the boxcar
    secondsL = [(c - liUTC[0]).total_seconds() for c in liUTC]
        
    # Boxcar width should equal the time constant
    #wid = 0.5 * (boxcarWidth / binSize)
    #sigma = (wid*binSize) / 6
    sigma = boxcarWidth / 8
    
    gausAv = []
    for sec, i in zip(secondsL, var):
        boxcarStart = sec - (boxcarWidth/2)
        boxcarEnd = sec + (boxcarWidth/2)
        
        # Make sure the boxcar stays bounded on data edges
        if boxcarStart < 0:
            boxcarStart = 0
        if boxcarEnd > np.nanmax(secondsL):
            boxcarEnd = np.nanmax(secondsL)
        
        binVar = [var[idx] for idx, d in enumerate(secondsL) if boxcarStart <= d <= boxcarEnd]
        binDist = [sec-d for idx, d in enumerate(secondsL) if boxcarStart <= d <= boxcarEnd]
        binDist = np.asarray(binDist)
        
        mu = 0
        weights = (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((binDist - mu) / sigma) ** 2)
        weightAvg = np.nansum(binVar*weights) / np.nansum(weights[~np.isnan(binVar)])
        
        # Will replace with a nan value if distance to nearest real value is 
        # more than the 2*sampling rate
        if not list(binDist[~np.isnan(binVar)]):
            weightAvg = np.nan
        elif np.nanmin(abs(binDist[~np.isnan(binVar)]))>(2*binSize):
            weightAvg = np.nan
            
        gausAv.append(weightAvg)
    return gausAv

# Finds map extent for radar plotting
def radarBounds(lflats, lflons, clat, clon, rlat, rlon):
    dminlat = np.nanmean(lflats) - 0.08
    dmaxlat = np.nanmean(lflats) + 0.08
    dminlon = np.nanmean(lflons) - 0.08
    dmaxlon = np.nanmean(lflons) + 0.08
    
    maxlon = np.nanmax([dmaxlon, rlon, clon])
    minlon = np.nanmin([dminlon, rlon, clon])
    maxlat = np.nanmax([dmaxlat, rlat, clat])
    minlat = np.nanmin([dminlat, rlat, clat])
        
    return maxlat, minlat, maxlon, minlon


# Will replace nan values into an observation array given a time range
class TimeFilter:
    def __init__(self, start, end):
        self.start_time = start
        self.end_time = end
        
    def insertNan(self, var, UTC):
        UTC = np.asarray(UTC)
        mask = (UTC >= self.start_time) & (UTC <= self.end_time)
        var[mask] = np.nan
        return var

def rhiBdDistance(radar, caseID, ts, azi):
    
    rhiscantime=datetime.datetime.strptime(radar.time['units'][14:-1], "%Y-%m-%dT%H:%M:%S")

    # Use the radar coordinates as the start of the RHI ray
    lat1 = radar.latitude['data'][0]
    lon1 = radar.longitude['data'][0]

    # Grab the end lat/lon of the RHI ray
    lat2,lon2 = getLocation(radar.latitude['data'][0], radar.longitude['data'][0], azi, 18)
    
    # High res the RHI ray
    hrlats, hrlons = highBdCoords([lat1, lat2], [lon1, lon2])

    # Find bdcoord for the RHI ray
    bdcoordFull, bidxFull = boundaryRelative(hrlats, hrlons, [rhiscantime]*len(hrlats), ts, caseID)
    
    # Nearest value on the boundary
    midx = np.argmin(np.abs(bdcoordFull))
    
    # Grab all the hres boundary coords
    orient = torient(ts, caseID)
    bdlat, bdlon = newBdCoords(rhiscantime, caseID)
    bdlatH, bdlonH = highBdCoords(bdlat, bdlon)
    
    # Find the nearest RHI ray coord to the boundary
    bdlatR = hrlats[midx]
    bdlonR = hrlons[midx]
    
    # Find the actual boundary coordinate
    bdlatClose = bdlatH[bidxFull[midx]]
    bdlonClose = bdlonH[bidxFull[midx]]
    
    # Find the kabdrelcoord using the location on the boundary
    kabdRelCoord = np.asarray([getDistance(lat1, lon1, bdlatClose, bdlonClose)])
    
    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 8))
    ax1.scatter(hrlons, hrlats)
    ax1.scatter(lon1, lat1, color='red')
    ax1.scatter(lon2, lat2, color='red')
    ax1.scatter(bdlonH, bdlatH, color='black')
    ax1.scatter(bdlonR, bdlatR, color='green', zorder=3)
    ax1.scatter(bdlonClose, bdlatClose, color='red', zorder=3)
    ax1.text(bdlonClose, bdlatClose, s=f'RHI scan distance: {round(kabdRelCoord[0], 2)}', fontsize=20, color='gray')
    plt.show()
    
    return kabdRelCoord