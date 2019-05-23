import math
import numpy as np


# function for creating decision vector based on antigen value
# at a specific concentration
def run_compare(df, analyte_val):
    above, below, llq, ulq, na = False, False, False, False, False
    value = df[analyte_val]
    try:
        float_val = float(value)
        if math.isnan(float_val):
            na = True
        elif float_val > THRESHOLDS[analyte_val]:
            above = True
        elif float_val < THRESHOLDS[analyte_val]:
            below = True
    except ValueError:
        if '<' in value:
            llq = True
        elif '>' in value:
            ulq = True
    finally:
        return np.array([above, below, llq, ulq, na])


# create decision matrices for determining which concentration to use
HRP2_MATRIX = np.array([['1:50', '1:50', '1:50', '1:50', '1:50'],
                        ['1:50', 'neat', 'neat', '1:50', 'fail'],
                        ['1:50', 'neat', 'neat', 'fail', 'fail'],
                        ['1:50', '1:50', 'fail', '1:50', '1:50'],
                        ['fail', '1:50', '1:50', 'fail', 'fail']])

LDH_PAN_MATRIX = np.array([['1:50', 'neat', 'neat', '1:50', '1:50'],
                           ['1:50', 'neat', 'neat', '1:50', 'fail'],
                           ['1:50', 'neat', 'neat', 'fail', 'fail'],
                           ['1:50', 'neat', 'fail', '1:50', '1:50'],
                           ['fail', 'neat', 'neat', 'fail', 'fail']])

LDH_PV_MATRIX = np.array([['1:50', 'neat', 'neat', '1:50', '1:50'],
                          ['1:50', 'neat', 'neat', '1:50', 'fail'],
                          ['1:50', 'neat', 'neat', 'fail', 'fail'],
                          ['1:50', 'neat', 'fail', '1:50', '1:50'],
                          ['fail', 'neat', 'neat', 'fail', 'fail']])

CRP_MATRIX = np.array([['1:50', 'neat', 'neat', '1:50', '1:50'],
                       ['1:50', 'neat', 'neat', '1:50', 'fail'],
                       ['1:50', 'neat', 'neat', 'fail', 'fail'],
                       ['1:50', 'neat', 'fail', '1:50', '1:50'],
                       ['fail', 'neat', 'neat', 'fail', 'fail']])

# decisions for various analytes
DECISIONS = {'HRP2_pg_ml': HRP2_MATRIX, 'LDH_Pan_pg_ml': LDH_PAN_MATRIX,
             'LDH_Pv_pg_ml': LDH_PV_MATRIX, 'CRP_ng_ml': CRP_MATRIX}

# threshhold values for various analytes
THRESHOLDS = {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514,
              'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574}

# positivity threshholds for various analytes
POS_THRESHOLDS = {'HRP2_pg_ml': 2.3, 'LDH_Pan_pg_ml': 47.8,
                  'LDH_Pv_pg_ml': 75.1, 'CRP_ng_ml': np.nan}