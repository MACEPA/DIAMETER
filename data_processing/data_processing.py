import os
import pandas as pd
import numpy as np
from functools import partial, reduce
# import helper function
from data_processing.data_processing_helpers import run_compare, return_decisions, fix_concentrations
# import constants
from data_processing.data_processing_helpers import THRESHOLDS


def main():
    dfs = []
    input_path = 'C:/Users/lzoeckler/Desktop/4plex/input_data/20190610'
    for fname in os.listdir(input_path):
        plex_data = pd.read_csv('{}/{}'.format(input_path, fname),
                                skiprows=8, names=['patient_id', 'type', 'well', 'error',
                                                   'HRP2_pg_ml', 'LDH_Pan_pg_ml',
                                                   'LDH_Pv_pg_ml', 'CRP_ng_ml'])
        plex_data = plex_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        plex_data['patient_id'] = plex_data['patient_id'].fillna(method='ffill')
        plex_data = plex_data[~plex_data['patient_id'].isnull()]
        dfs.append(plex_data)
    samples_data = pd.concat(dfs)
    # subset data to just what we want
    samples_data = samples_data.loc[~samples_data['type'].isnull()]
    samples_data = samples_data.loc[~samples_data['type'].str.contains('pixel')]
    samples_data = samples_data.loc[samples_data['patient_id'].str.contains('pa-')]
    samples_data = samples_data.drop('type', axis=1)
    # break out concentratino from patient string
    samples_data['concentration'] = samples_data['patient_id'].apply(lambda x: x.partition(' ')[-1])
    samples_data['patient_id'] = samples_data['patient_id'].apply(lambda x: x.partition(' ')[0])
    samples_data = samples_data.loc[
        (samples_data['concentration'].str.contains('neat|50|2500|125000|6250000|312500000'))]
    samples_data = samples_data.loc[~samples_data['concentration'].str.contains('low volume')]
    samples_data = samples_data.loc[~samples_data['well'].isnull()]
    # fix concentrations
    samples_data['concentration'] = samples_data.apply(fix_concentrations, axis=1)
    samples_data = samples_data.sort_values(['patient_id', 'well'])

    # generate an empty list to fill with small dfs, which will be combined
    analyte_dfs = []
    # run counts for decision on what to keep
    for analyte in THRESHOLDS.keys():
        patient_dfs = []
        # iterate over patient_ids
        for i in samples_data['patient_id'].unique():
            patient_data = samples_data.loc[samples_data['patient_id'] == i]
            # get number of dilutions
            dilution_values = sorted([val for val in patient_data['concentration'].unique() if val != '1'], key=len)
            # set initial best decision to neat (1)
            best_decision = '1'
            # iterate over dilution values
            for max_dilution in dilution_values:
                # subset to dilutions
                dil_data = patient_data.loc[patient_data['concentration'].isin([best_decision, max_dilution])]
                # create partial function for generating decision vectors
                partial_compare = partial(run_compare, analyte_val=analyte, dil_val=max_dilution)
                # generate decision vectors
                dil_data['decision_vector'] = dil_data.apply(partial_compare, axis=1)
                # pull decision matrix for given analyte and concentrations
                decisions = return_decisions(best_decision, max_dilution)
                decision_matrix = decisions[analyte]
                # construct empty dataframe to hold best values
                best_df = pd.DataFrame(columns=['patient_id', analyte,
                                                '{}_dilution'.format(analyte),
                                                '{}_well'.format(analyte)])
                try:

                    vector_low = dil_data.loc[dil_data['concentration'] == best_decision,
                                              'decision_vector'].item()
                    vector_high = dil_data.loc[dil_data['concentration'] == max_dilution,
                                               'decision_vector'].item()
                    decision = decision_matrix[vector_high, vector_low].item()
                    if decision in [best_decision, max_dilution]:
                        val = dil_data.loc[dil_data['concentration'] == decision,
                                           analyte].item()
                        well = dil_data.loc[dil_data['concentration'] == decision,
                                            'well'].item()
                        best_decision = decision
                    elif decision == 'fail':
                        val = np.nan
                        well = np.nan
                    else:
                        raise ValueError("Unexpected decision value: {}".format(decision))
                    best_df = best_df.append({'patient_id': i, analyte: val,
                                              '{}_dilution'.format(analyte): decision,
                                              '{}_well'.format(analyte): well}, ignore_index=True)
                    best_decision = decision
                    if decision == 'fail':
                        break
                except ValueError:
                    print("ValueError:", analyte, max_dilution, i)
            patient_dfs.append(best_df)
        patient_df = pd.concat(patient_dfs)
        analyte_dfs.append(patient_df)
    output_df = reduce(lambda left, right: pd.merge(left, right, on='patient_id'), analyte_dfs)
    output_df.to_csv('C:/Users/lzoeckler/Desktop/4plex/output_data/final_dilutions.csv', index=False)
    return output_df


if __name__ == '__main__':
    main()
