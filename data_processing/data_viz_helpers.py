import numpy as np
import matplotlib.cm as cm


# function for removing > and < from strings so they can be cast to float
def clean_strings(val):
    clean = val.replace('<', '')
    clean = clean.replace('>', '')
    if clean == 'fail':
        return np.nan
    return clean


# get all colors and shapes for association
all_colors = cm.rainbow(np.linspace(0, 1, 8))
all_dilutions = ['1', '50', '2500', '125000', '6250000', '312500000', '15625000000',
                 '781250000000']
all_shapes = ['+', 'v', 's', 'p', 'd', '^', '.', '*']

# associate colors to different dilution values
combo = zip(all_dilutions, all_colors)
COLOR_DICT = {dil: val for dil, val in combo}
COLOR_DICT['fail'] = np.array([0.0, 0.0, 0.0, 0.0])

# associate shapes to different dilution values
shape_combo = zip(all_dilutions, all_shapes)
SHAPE_DICT = {dil: val for dil, val in shape_combo}

# list of all analytes and assoicated name components
ANALYTE_INFO = {'HRP2_pg_ml': ('HRP2', 'pg/ml'), 'LDH_Pan_pg_ml': ('LDH_Pan', 'pg/ml'),
                'LDH_Pv_pg_ml': ('LDH_Pv', 'pg/ml'), 'CRP_ng_ml': ('CRP', 'ng/ml')}
