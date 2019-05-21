import pandas as pd
import numpy as np
import math
from functools import partial, reduce


# function for creating decision vector based on antigen value
# at a specific concentration
def run_compare(df, analyte_val):
    above, below, llq, ulq, na = False, False, False, False, False
    value = df[analyte_val]
    try:
        float_val = float(value)
        if math.isnan(float_val):
            na = True
        elif float_val > threshholds[analyte_val]:
            above = True
        elif float_val < threshholds[analyte_val]:
            below = True
    except ValueError:
        if '<' in value:
            llq = True
        elif '>' in value:
            ulq = True
    finally:
        return np.array([above, below, llq, ulq, na])


# create decision matrices for determining which concentration to use
HRP2_matrix = np.array([['1:50', '1:50', '1:50', '1:50', '1:50'],
                        ['1:50', 'neat', 'neat', '1:50', 'fail'],
                        ['1:50', 'neat', 'neat', 'fail', 'fail'],
                        ['1:50', '1:50', 'fail', '1:50', '1:50'],
                        ['fail', '1:50', '1:50', 'fail', 'fail']])

LDH_Pan_matrix = np.array([['1:50', 'neat', 'neat', '1:50', '1:50'],
                           ['1:50', 'neat', 'neat', '1:50', 'fail'],
                           ['1:50', 'neat', 'neat', 'fail', 'fail'],
                           ['1:50', 'neat', 'fail', '1:50', '1:50'],
                           ['fail', 'neat', 'neat', 'fail', 'fail']])

LDH_Pv_matrix = np.array([['1:50', 'neat', 'neat', '1:50', '1:50'],
                          ['1:50', 'neat', 'neat', '1:50', 'fail'],
                          ['1:50', 'neat', 'neat', 'fail', 'fail'],
                          ['1:50', 'neat', 'fail', '1:50', '1:50'],
                          ['fail', 'neat', 'neat', 'fail', 'fail']])

CRP_matrix = np.array([['1:50', 'neat', 'neat', '1:50', '1:50'],
                       ['1:50', 'neat', 'neat', '1:50', 'fail'],
                       ['1:50', 'neat', 'neat', 'fail', 'fail'],
                       ['1:50', 'neat', 'fail', '1:50', '1:50'],
                       ['fail', 'neat', 'neat', 'fail', 'fail']])

# decisions for various analytes
decisions = {'HRP2_pg_ml': HRP2_matrix, 'LDH_Pan_pg_ml': LDH_Pan_matrix,
             'LDH_Pv_pg_ml': LDH_Pv_matrix, 'CRP_ng_ml': CRP_matrix}

# threshhold values for various analytes
threshholds = {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514,
               'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574}

# positivity threshholds for various analytes
pos_threshholds = {'HRP2_pg_ml': 2.3, 'LDH_Pan_pg_ml': 47.8,
                   'LDH_Pv_pg_ml': 75.1, 'CRP_ng_ml': np.nan}

for fname in ["file", "paths", "here"]:
    # read in data from flat file, columns must be in correct order
    plex_data = pd.read_csv(fname, skiprows=8, names=['patient_id', 'type',
                                                      'well', 'error',
                                                      'HRP2_pg_ml',
                                                      'LDH_Pan_pg_ml',
                                                      'LDH_Pv_pg_ml',
                                                      'CRP_ng_ml'])
    plex_data = plex_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
    plex_data['patient_id'] = plex_data['patient_id'].fillna(method='ffill')
    # subset data to just what we want
    samples_data = plex_data.loc[plex_data['patient_id'].str.contains('pa-')]
    samples_data = samples_data.drop('type', axis=1)
    # rename patient_id and create concentration column
    samples_data['concentration'] = samples_data['patient_id'].apply(
        lambda x: x.partition(' ')[-1])
    samples_data['patient_id'] = samples_data['patient_id'].apply(
        lambda x: x.partition(' ')[0])
    # subset data a final time
    samples_data = samples_data.loc[
        (samples_data['concentration'] == '(neat)') |
        (samples_data['concentration'].str.contains('50x'))]
    samples_data = samples_data.loc[~samples_data[
        'concentration'].str.contains('low volume')]

    # generate an empty list to fill with small dfs, which will be combined
    final_dfs = []
    # run counts for decision on what to keep
    for analyte in threshholds.keys():
        # for analyte in ['HRP2_pg_ml']:
        # create partial function for generating decision vectors
        partial_compare = partial(run_compare, analyte_val=analyte)
        # generate decision vectors
        samples_data['decision_vector'] = samples_data.apply(partial_compare, axis=1)
        # pull decision matrix for given analyte
        decision_matrix = decisions[analyte]
        # generate an empty list to fill with tiny dfs, which will be combined
        tiny_dfs = []
        # iterate over patient_ids
        for i in samples_data['patient_id'].unique().tolist():
            tiny_df = pd.DataFrame(columns=['patient_id', analyte,
                                            '{}_dilution'.format(analyte),
                                            '{}_well'.format(analyte)])
            sub_data = samples_data.loc[samples_data['patient_id'] == i]
            neat_vector = sub_data.loc[sub_data['concentration'] == '(neat)',
                                       'decision_vector'].item()
            dil_vector = sub_data.loc[sub_data['concentration'].str.contains('50x'),
                                      'decision_vector'].item()
            decision = decision_matrix[neat_vector, dil_vector].item()
            pos_val = 'negative'
            if decision == '1:50':
                val = sub_data.loc[sub_data['concentration'].str.contains('50x'),
                                   analyte].item()
                well = sub_data.loc[sub_data['concentration'].str.contains('50x'),
                                    'well'].item()
            elif decision == 'neat':
                val = sub_data.loc[sub_data['concentration'] == '(neat)',
                                   analyte].item()
                well = sub_data.loc[sub_data['concentration'] == '(neat)',
                                    'well'].item()
            elif decision == 'fail':
                val = np.nan
                well = np.nan
            else:
                raise ValueError("Unexpected decision value: {}".format(decision))
            tiny_df = tiny_df.append({'patient_id': i, analyte: val,
                                      '{}_dilution'.format(analyte): decision,
                                      '{}_well'.format(analyte): well},
                                     ignore_index=True)
            tiny_dfs.append(tiny_df)
        small_df = pd.concat(tiny_dfs)
        final_dfs.append(small_df)
    output_df = reduce(lambda left, right: pd.merge(left, right, on='patient_id'),
                       final_dfs)
