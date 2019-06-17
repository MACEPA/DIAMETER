import numpy as np


# function for removing > and < from strings so they can be cast to float
def clean_strings(val):
    clean = val.replace('<', '')
    clean = clean.replace('>', '')
    if clean == 'fail':
        return np.nan
    return clean
