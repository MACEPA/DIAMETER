import math
import numpy as np


# function for creating decision vector based on antigen value
# at a specific concentration
def run_compare(df, dil_constants, analyte_val, dil_val):
    above, below, llq, ulq, na = False, False, False, False, False
    val = df[analyte_val]
    thresh_val = dil_constants[dil_val] * THRESHOLDS[analyte_val]
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


# function for removing 'day' from strings
def remove_day(x):
    if isinstance(x, str):
        x = x.replace('day ', '')
    return x


# generate dilution constants based on initial dilution value
def build_dil_constants(base_dil):
    return {str(base_dil**i): (base_dil**(i-1)) for i in range(1, 10)


# generate dilution sets based on initial dilution value
def build_dil_sets(base_dil):
    return {str(base_dil ** i): (str(base_dil ** (i - 1)), str(base_dil ** 1), 'fail') for i in range(1, 10)}


# threshhold values for various analytes
THRESHOLDS = {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514,
              'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574}
