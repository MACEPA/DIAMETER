import numpy as np


# Function for cleaning analyte values
# This function DOES NOT log values
def clean_strings(val):
    if isinstance(val, str):
        clean = val.replace('<', '')
        clean = clean.replace('>', '')
        try:
            return float(clean)
        except ValueError:
            return np.nan
    elif isinstance(val, float) or isinstance(val, int):
        return float(val)
    else:
        return np.nan
