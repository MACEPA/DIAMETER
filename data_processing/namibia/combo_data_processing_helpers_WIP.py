import math
import numpy as np
import pandas as pd


def read_data(input_path, fname, plex):
    if plex == 4:
        plex_data = pd.read_csv('{}/{}'.format(input_path, fname), index_col=False,
                                skiprows=8, names=['patient_id', 'type', 'well', 'error',
                                                   'HRP2_pg_ml', 'LDH_Pan_pg_ml',
                                                   'LDH_Pv_pg_ml', 'CRP_ng_ml',
                                                   'fail1', 'fail2'])
        # certain CSVs have empty extra columns when read in for some reason
        # they need to be labeled and then dropped
        plex_data.drop(['fail1', 'fail2'], axis=1, inplace=True)
    elif plex == 5:
        plex_data = pd.read_csv('{}/{}'.format(input_path, fname),
                                skiprows=13, names=['patient_id', 'type', 'well', 'error',
                                                    'HRP2_pg_ml', 'LDH_Pan_pg_ml',
                                                    'LDH_Pv_pg_ml', 'LDH_Pf_pg_ml',
                                                    'CRP_ng_ml'])
    else:
        raise ValueError("Unexpected plex value: {}".format(plex))
    return plex_data


# function for creating decision vector based on antigen value
# at a specific concentration
def run_compare(df, analyte_val, dil_val, base, plex):
    above, below, llq, ulq, na = False, False, False, False, False
    val = df[analyte_val]
    dil_constants = build_dil_constants(base)
    thresh_val = dil_constants[dil_val] * THRESHOLDS[plex][analyte_val]
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
                 'LDH_Pv_pg_ml': other_matrix, 'LDH_Pf_pg_ml': other_matrix,
                 'CRP_ng_ml': other_matrix}
    return decisions


# function for splitting off time from patient_id string
def split_time(df):
    try:
        sub = df['patient_id'].split('-')
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
    return {str(base_dil**i): (base_dil**(i-1)) for i in range(1, 10)}


# generate dilution sets based on initial dilution value
def build_dil_sets(base_dil):
    return {str(base_dil**i): (str(base_dil**(i-1)), str(base_dil**1), 'fail') for i in range(1, 10)}


# threshhold values for various analytes, 4plex and 5plex
THRESHOLDS = {4: {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514, 'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574},
              5: {'HRP2_pg_ml': 2800, 'LDH_Pan_pg_ml': 67000, 'LDH_Pv_pg_ml': 19200, 'LDH_Pf_pg_ml': 20800,
                  'CRP_ng_ml': 38000}}
