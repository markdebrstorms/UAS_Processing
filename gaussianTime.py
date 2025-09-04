# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 10:49:39 2025

@author: mdebruin2
"""

import numpy as np
import datetime

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
    # Sigma controls the weight distribution away from center
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
