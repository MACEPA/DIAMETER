import math
import numpy as np
import pandas as pd
# import contants
from core_processing_helpers import THRESHOLDS


# function for creating decision vector based on antigen value
# at a specific concentration, 5plex
def run_compare(df, analyte_val, dil_val):
    # Columns = neat: [LLQ, real #, ULQ or within 20% ULQ, NA]
    # Rows = dilution: [LLQ or within 20x LLQ, real #, ULQ or within 20% ULQ, NA]
    llq, real, ulq, na = False, False, False, False
    val = df[analyte_val]
    ulq_val = int(dil_val) * thresholds['ulq'][analyte_val]
    llq_val = int(dil_val) * thresholds['llq'][analyte_val]
    # set decision vector based on value (order of operations is important here)
    try:
        float_val = float(val)
        if math.isnan(float_val):
            na = True
        elif (dil_val == '20') and (float_val < 20*llq_val):
            llq = True
        elif (dil_val == '1') and (float_val > (.8*ulq_val)):
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
    # subset dataframe to only real neat HRP2 values above 100 and below 80% of the ULQ,
    # and real 20x values below 80% of the 20x ULQ
    df = df.loc[~df['HRP2_pg_ml_neat_val'].str.contains('>')]
    df = df.loc[~df['HRP2_pg_ml_neat_val'].str.contains('<')]
    df = df.loc[~df['HRP2_pg_ml_20x_val'].str.contains('>')]
    df = df.loc[~df['HRP2_pg_ml_20x_val'].str.contains('<')]
    df = df.loc[df['HRP2_pg_ml'] != 'fail']
    df = df.loc[df['HRP2_pg_ml_neat_val'].astype(float) > 100]
    df = df.loc[df['HRP2_pg_ml_neat_val'].astype(float) < (.8*2800)]
    df = df.loc[df['HRP2_pg_ml_20x_val'].astype(float) < (.8*20*2800)]
    # subet further to either:
    # 1) 20x HRP2 values ULQ or
    alert_ulq = df.loc[df['HRP2_pg_ml_20x_val'].str.contains('>')]
    # 2) 20x HRP2 values greater than 8x neat HRP2 values
    alert_10x = df.loc[~df['HRP2_pg_ml_20x_val'].str.contains('>')]
    alert_10x = alert_10x.loc[~alert_10x['HRP2_pg_ml_20x_val'].str.contains('<')]
    alert_10x = alert_10x.loc[alert_10x['HRP2_pg_ml_20x_val'].astype(float) >
                              (alert_10x['HRP2_pg_ml_neat_val'].astype(float) * 8)]
    # combine options 1 and 2
    df = pd.concat([alert_ulq, alert_10x])
    # set selected dilution and other info to "alert"
    df['HRP2_pg_ml'] = 'alert'
    df['HRP2_pg_ml_dilution'] = np.nan
    df['HRP2_pg_ml_well'] = 'alert'
    return df


def return_decisions(low, high, fail='fail'):
    # Columns = neat: [LLQ, real #, ULQ or within 20% ULQ, NA]
    # Rows = dilution: [LLQ or within 20x LLQ, real #, ULQ or within 20% ULQ, NA]
    other_matrix = np.array([[low, low, fail, fail],
                             [low, low, high, fail],
                             [fail, fail, high, fail],
                             [fail, fail, high, fail]])
    # decisions for various analytes
    decisions = {'HRP2_pg_ml': other_matrix, 'LDH_Pan_pg_ml': other_matrix,
                 'LDH_Pv_pg_ml': other_matrix, 'LDH_Pf_pg_ml': other_matrix,
                 'CRP_ng_ml': other_matrix}
    return decisions
