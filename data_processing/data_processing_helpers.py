import math
import numpy as np


# function for creating decision vector based on antigen value
# at a specific concentration
def run_compare(df, analyte_val, dil_val):
    above, below, llq, ulq, na = False, False, False, False, False
    val = df[analyte_val]
    thresh_val = DIL_CONSTANTS[dil_val] * THRESHOLDS[analyte_val]
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


# function for cleaning concentration info
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
    hrp2_matrix = np.array([[high, high, high, high, high],
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
    decisions = {'HRP2_pg_ml': hrp2_matrix, 'LDH_Pan_pg_ml': other_matrix,
                 'LDH_Pv_pg_ml': other_matrix, 'CRP_ng_ml': other_matrix}
    return decisions


# function for splitting off time from patient_id string
def split_time(df):
    sub = df['patient_id'].split('-')
    try:
        time = int(sub[2])
        return time
    except IndexError:
        return 0


# function for removing time from patient_id string once it's split
def remove_time(df):
    patient = df['patient_id'].split('-')
    return '{}-{}'.format(patient[0], patient[1])


# threshhold values for various analytes
THRESHOLDS = {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514,
              'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574}

# positivity threshholds for various analytes
POS_THRESHOLDS = {'HRP2_pg_ml': 2.3, 'LDH_Pan_pg_ml': 47.8,
                  'LDH_Pv_pg_ml': 75.1, 'CRP_ng_ml': np.nan}

# constant to apply to the threshhold for different dilutions
DIL_CONSTANTS = {'50': 1, '2500': 50, '125000': 2500,
                 '6250000': 125000, '312500000': 6250000,
                 '15625000000': 312500000, '781250000000': 15625000000}

# dilution sets for various dilutions
DILUTION_SETS = {'50': ('1', '50', 'fail'), '2500': ('50', '2500', 'fail'),
                 '125000': ('2500', '125000', 'fail'),
                 '6250000': ('125000', '6250000', 'fail'),
                 '312500000': ('6250000', '312500000', 'fail'),
                 '15625000000': ('312500000', '15625000000', 'fail'),
                 '781250000000': ('15625000000', '781250000000', 'fail')}
