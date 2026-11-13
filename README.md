## Description of UAS_Processing for De Bruin et al. 2026 and Houston et al. 2026. 
Houston et al. 2026 describes the processing of UAS data collected during TORUS 2019 and TORUS LItE 2023. The functions used for correcting the thermodynamic sensors are contained in this repository. De Bruin et al. 2026 is the application of the processed UAS data, as well as radar and mobile mesonet data, for analysis of supercell left flank boundaries. All code used in this repository is Python.

## Contents
UASProcessingFunctions.py contains the code used from the the master's thesis of Mark De Bruin which processed and analyzed UAS/radar/mobile mesonet data. Note that while the script contains most of the functions used for the processing and analysis of the data, it does not include the data itself or the extra scripting needed to call each function. The file also contains descriptions for the critical functions. A few of the critical functions for this work include: deficit, boundaryRelative, hvort2, hvortR, vvort, divergence, and bdSmooth.

RHcorrection.py corrects the iPTH sensor RH using an empirically derived bias correction function.

time_response_correct.py is used for correcting the time lag of the MHP sensor in 2019 and the iPTH sensor in 2023. hysteresisCorrection.ipynb applies and visualizes the function.

gaussianTime.py is the gaussian weighted rolling boxcar smoother (bin-based averaging) used to smooth the inputs to the time response correction and the inputs to the objective analysis function in De Bruin et al. 2026.
