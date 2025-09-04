# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 12:39:24 2025

@author: mdebruin2
"""

import numpy as np
import datetime


# Function for hysteresis corrections.
# Equation is Var = tau*(dvar/dt)+ var_hat
# var = corrected temp or RH
# tau = time response
# dvar/dt = centered differencing (using previous measurement and next measurement)
# var_hat = actual observed measurement
# t = list of datetime objects
def time_response_correct(var_hat, t, tau):
    var = [np.nan]
    for i in range(1,np.asarray(var_hat).shape[0] - 1):
        t_diff = (t[i+1] - t[i-1]).total_seconds()
        var_new = (tau*((var_hat[i+1]-var_hat[i-1])/(1*t_diff))) + var_hat[i]
        var.append(var_new)

    # Centered difference means index 0/-1 can't be solved for    
    var.append(np.nan)
    var_corr = np.asarray(var)
    return var_corr