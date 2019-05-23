import pandas as pd
import numpy as np
from functools import partial, reduce
# import helper function
from data_processing.data_processing_helpers import run_compare
# import constants
from data_processing.data_processing_helpers import (DECISIONS,
                                                     THRESHOLDS)


for fname in ["file", "paths", "here"]:
    # read in data from flat file, columns must be in correct order
    plex_data = pd.read_csv(fname, skiprows=8, names=['patient_id', 'type',
                                                      'well', 'error',
                                                      'HRP2_pg_ml',
                                                      'LDH_Pan_pg_ml',
                                                      'LDH_Pv_pg_ml',
                                                      'CRP_ng_ml'])
    plex_data = plex_data.applymap(lambda x: x.lower() if isinstance(
        x, str) else x)
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
    for analyte in THRESHOLDS.keys():
        # create partial function for generating decision vectors
        partial_compare = partial(run_compare, analyte_val=analyte)
        # generate decision vectors
        samples_data['decision_vector'] = samples_data.apply(partial_compare, axis=1)
        # pull decision matrix for given analyte
        decision_matrix = DECISIONS[analyte]
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
