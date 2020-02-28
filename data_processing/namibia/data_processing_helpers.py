import math
import numpy as np
import pandas as pd


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


# function for creating decision vector based on antigen value
# at a specific concentration, 5plex
def run_5plex_compare(df, analyte_val, dil_val):
    # Columns = neat: [LLQ, real #, ULQ or within 20% ULQ, NA]
    # Rows = dilution: [LLQ or within 20x LLQ, real #, ULQ or within 20% ULQ, NA]
    llq, real, ulq, na = False, False, False, False
    val = df[analyte_val]
    ulq_val = int(dil_val) * THRESHOLDS[5]['ulq'][analyte_val]
    llq_val = int(dil_val) * THRESHOLDS[5]['llq'][analyte_val]
    # set truth vector based on value (order is important)
    try:
        float_val = float(val)
        if math.isnan(float_val):
            na = True
        elif (dil_val == '20') and (float_val < 20*llq_val):
            llq = True
        elif float_val > (.8*ulq_val):
            ulq = True
        else:
            real = True
    except ValueError:
        if '<' in val:
            llq = True
        elif '>' in val:
            ulq = True
        else:
            raise ValueError("Unexpected value: {}".format(val))
    finally:
        return np.array([llq, real, ulq, na])


# function for setting HRP2 hook effect alerts
def set_hrp2_alerts(df):
    # subset dataframe to only real neat HRP2 values below 100
    df = df.loc[~df['HRP2_pg_ml_neat_val'].str.contains('>')]
    df = df.loc[~df['HRP2_pg_ml_neat_val'].str.contains('<')]
    df = df.loc[df['HRP2_pg_ml_neat_val'] != 'fail']
    df = df.loc[df['HRP2_pg_ml_neat_val'].astype(float) < 100]
    # subet further to either:
    # 1) 20x HRP2 values ULQ or
    alert_ulq = df.loc[df['HRP2_pg_ml_20x_val'].str.contains('>')]
    # 2) 20x HRP2 values greater than 10x neat HRP2 values
    alert_10x = df.loc[~df['HRP2_pg_ml_20x_val'].str.contains('>')]
    alert_10x = alert_10x.loc[~alert_10x['HRP2_pg_ml_20x_val'].str.contains('<')]
    alert_10x = alert_10x.loc[alert_10x['HRP2_pg_ml_20x_val'].astype(float) >
                              (alert_10x['HRP2_pg_ml_neat_val'].astype(float) * 10)]
    # combine options 1 and 2
    df = pd.concat([alert_ulq, alert_10x])
    # set selected dilution and other info to "alert"
    df['HRP2_pg_ml'] = 'alert'
    df['HRP2_pg_ml_dilution'] = np.nan
    df['HRP2_pg_ml_well'] = 'alert'
    return df


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


def return_5plex_decisions(low, high, fail='fail'):
    # Columns = neat: [LLQ, real #, ULQ or within 20% ULQ, NA]
    # Rows = dilution: [LLQ or within 20x LLQ, real #, ULQ or within 20% ULQ, NA]
    other_matrix = np.array([[low, low, fail, fail],
                             [low, low, high, fail],
                             [fail, fail, high, fail],
                             [fail, fail, fail, fail]])
    # decisions for various analytes
    decisions = {'HRP2_pg_ml': other_matrix, 'LDH_Pan_pg_ml': other_matrix,
                 'LDH_Pv_pg_ml': other_matrix, 'LDH_Pf_pg_ml': other_matrix,
                 'CRP_ng_ml': other_matrix}
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
THRESHOLDS = {4: {'ulq': {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514, 'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574},
                  'llq': {}},
              5: {'ulq': {'HRP2_pg_ml': 2800, 'LDH_Pan_pg_ml': 67000, 'LDH_Pv_pg_ml': 19200, 'LDH_Pf_pg_ml': 20800,
                          'CRP_ng_ml': 38000},
                  'llq': {'HRP2_pg_ml': .68, 'LDH_Pan_pg_ml': 16.36, 'LDH_Pv_pg_ml': 4.96, 'LDH_Pf_pg_ml': 5.08,
                          'CRP_ng_ml': 9.28}}}