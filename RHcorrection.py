import numpy as np

# Corrects iPTH RH using raw temp and RH
# Inputs: rh = uncorrected RH (%), temp = fast temperature
# Outputs: rhcA = array of corrected RH (bias and hysteresis)
# Will clip to 99.9% for raw RH >94%
def iprhCorrect(rh, temp):
    rhcA = np.asarray([])
    for r, t in zip(rh, temp):
        rhc = (13.68 + (0.1244*r) + (-0.03791*t) + (0.0334*(r**2)) + (0.001814*r*t) + (-0.0003154*(r**3)) + (-6.892e-5*(r**2)*t) +
        (4.864e-7*(r**4)) + (5.87e-7*(r**3)*t))
        rhcA = np.append(rhcA, rhc)
    rhcA = np.clip(rhcA, 0, 99.9)
    return rhcA