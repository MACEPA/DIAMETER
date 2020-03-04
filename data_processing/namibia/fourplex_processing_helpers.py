import math
import numpy as np
# import contants
from core_processing_helpers import THRESHOLDS


# function for creating decision vector based on antigen value
# at a specific concentration, 4plex
def run_compare(df, dil_constants, analyte_val, dil_val):
    above, below, llq, ulq, na = False, False, False, False, False
    val = df[analyte_val]
    thresh_val = dil_constants[dil_val] * THRESHOLDS[4]['ulq'][analyte_val]
    # set truth vector based on value (order is important)
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


# function for removing 'day' from strings
def remove_day(x):
    if isinstance(x, str):
        x = x.replace('day ', '')
    return x
