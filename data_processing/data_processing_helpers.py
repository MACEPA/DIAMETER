import math
import numpy as np


# function for creating decision vector based on antigen value
# at a specific concentration
def run_compare(df, analyte_val, dil_constant):
    above, below, llq, ulq, na = False, False, False, False, False
    val = df[analyte_val]
    thresh_val = dil_constant * THRESHOLDS[analyte_val]
    try:
        float_val = float(val)
        if math.isnan(float_val):
            na = True
        elif float_val > thresh_val:
            above = True
        elif float_val < thresh_val:
            below = True
    except ValueError:
        if '<' in val:
            llq = True
        elif '>' in val:
            ulq = True
    finally:
        return np.array([above, below, llq, ulq, na])


# function for returning accurate concentrations from patient string
def fix_concentrations(df):
    con = df['concentration'].partition(':')[2]
    con = con.partition(')')[0]
    if len(con) != 0:
        return con
    else:
        return '1'


# create decision matrices for determining which concentration to use
def return_decisions(low, high, fail='fail'):
    # Columns = [neat_above, neat_below, neat_LLQ, neat_ULQ, NA]
    # Rows = [dil_above, dil_below, dil_LLQ, dil_ULQ, NA]
    HRP2_matrix = np.array([[high, high, high, high, high],
                            [high, low, low, high, fail],
                            [high, low, low, fail, fail],
                            [high, high, fail, high, high],
                            [fail, high, high, fail, fail]])

    other_matrix = np.array([[high, low, low, high, high],
                               [high, low, low, high, fail],
                               [high, low, low, fail, fail],
                               [high, low, fail, high, high],
                               [fail, low, low, fail, fail]])

    # decisions for various analytes
    decisions = {'HRP2_pg_ml': HRP2_matrix, 'LDH_Pan_pg_ml': other_matrix,
                 'LDH_Pv_pg_ml': other_matrix, 'CRP_ng_ml': other_matrix}
    return decisions


# threshhold values for various analytes
THRESHOLDS = {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514,
              'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574}

# positivity threshholds for various analytes
POS_THRESHOLDS = {'HRP2_pg_ml': 2.3, 'LDH_Pan_pg_ml': 47.8,
                  'LDH_Pv_pg_ml': 75.1, 'CRP_ng_ml': np.nan}

# constant to apply to the threshhold for different dilutions
DIL_CONSTANTS = {'50x': 1, '2500x': 50, '125000x': 2500,
                 '6250000x': 125000, '312500000x': 6250000}

# dilution sets for various dilutions
DILUTION_SETS = {'50x': ('neat', '50x', 'fail'), '2500x': ('50x', '2500x', 'fail'),
                 '125000x': ('2500x', '125000x', 'fail'),
                 '6250000x': ('125000x', '6250000x', 'fail'),
                 '312500000x': ('6250000x', '312500000x', 'fail')}
