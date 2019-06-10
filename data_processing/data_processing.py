import pandas as pd
import numpy as np
from functools import partial, reduce
# import helper function
from data_processing.data_processing_helpers import run_compare, return_decisions
# import constants
from data_processing.data_processing_helpers import DIL_CONSTANTS, DILUTION_SETS, THRESHOLDS


def main():
    dfs = []
    for fname in ['other_dilutions_input', 'neat_dilution_input']:
        # read in data from flat file, columns must be in correct order
        plex_data = pd.read_csv('C:/Users/lzoeckler/Desktop/4plex/test_data/{}.csv'.format(fname),
                                skiprows=8, names=['patient_id', 'type', 'well', 'error',
                                                   'HRP2_pg_ml', 'LDH_Pan_pg_ml',
                                                   'LDH_Pv_pg_ml', 'CRP_ng_ml'])
        plex_data = plex_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        plex_data['patient_id'] = plex_data['patient_id'].fillna(method='ffill')
        dfs.append(plex_data)
    samples_data = pd.concat(dfs)
    # subset data to just what we want
    samples_data = samples_data.loc[samples_data['patient_id'].str.contains('pa-')]
    samples_data = samples_data.drop('type', axis=1)
    # rename patient_id and create concentration column
    samples_data['concentration'] = samples_data['patient_id'].apply(
        lambda x: x.partition(' ')[-1])
    samples_data['patient_id'] = samples_data['patient_id'].apply(
        lambda x: x.partition(' ')[0])
    # subset data a final time
    subset_str = 'neat|50x|2500x|125000x|6250000x|312500000x'
    samples_data = samples_data.loc[(samples_data['concentration'].str.contains(subset_str))]
    samples_data = samples_data.loc[~samples_data['concentration'].str.contains('low volume')]
    samples_data = samples_data.loc[~samples_data['well'].isnull()]
    samples_data = samples_data.sort_values(['patient_id', 'well'])

    # generate an empty list to fill with small dfs, which will be combined
    final_dfs = []
    # run counts for decision on what to keep
    for analyte in THRESHOLDS.keys():
        dil_dfs = []
        for max_dilution in DIL_CONSTANTS.keys():
            # create partial function for generating decision vectors
            partial_compare = partial(run_compare, analyte_val=analyte)
            # get number of dilutions
            dilution_number = len(samples_data['concentration'].unique().tolist())
            # generate decision vectors
            samples_data['decision_vector'] = samples_data.apply(partial_compare, axis=1)
            # generate all decicion matrices given current concentrations
            low, high, fail = DILUTION_SETS[max_dilution]
            decisions = return_decisions(low, high, fail)
            # pull decision matrix for given analyte
            decision_matrix = decisions[analyte]
            # generate an empty list to fill with tiny dfs, which will be combined
            tiny_dfs = []
            # iterate over patient_ids
            for i in samples_data['patient_id'].unique().tolist():
                # create a dumby dataframe to fill later
                tiny_df = pd.DataFrame(columns=['patient_id', analyte,
                                                '{}_dilution'.format(analyte),
                                                '{}_well'.format(analyte)])
                tiny_df['comparison'] = '{} vs {}'.format(low, high)
                tiny_df = tiny_df[['patient_id', 'comparison', '{}_dilution'.format(analyte),
                                   '{}_well'.format(analyte)]]
                # subset data to current patient_id
                sub_data = samples_data.loc[samples_data['patient_id'] == i]
                # only run the decision process if all possible concentrations exist for the
                # current patient_id
                if len(sub_data) == dilution_number:
                    # return low concentration deicision vector
                    vector_low = sub_data.loc[sub_data['concentration'].str.contains(low),
                                              'decision_vector'].item()
                    # return high concentration decision vector
                    vector_high = sub_data.loc[sub_data['concentration'].str.contains(high),
                                               'decision_vector'].item()
                    # determine decision using low and high decision vectors
                    decision = decision_matrix[vector_low, vector_high].item()
                    # if the decision is one of the low or high concentrations, set val and
                    # well accordingly
                    if decision in [low, high]:
                        val = sub_data.loc[sub_data['concentration'].str.contains(decision),
                                           analyte].item()
                        well = sub_data.loc[sub_data['concentration'].str.contains(decision),
                                            'well'].item()
                    # if the decision is the fail case, set val and well to NaN
                    elif decision == fail:
                        val = np.nan
                        well = np.nan
                    # otherwise, something went wrong...
                    else:
                        raise ValueError("Unexpected decision value: {}".format(decision))
                    # append everything on and combine up
                    tiny_df = tiny_df.append({'patient_id': i,
                                              'comparison': '{} vs {}'.format(low, high),
                                              analyte: val, '{}_dilution'.format(analyte): decision,
                                              '{}_well'.format(analyte): well}, ignore_index=True)
                    tiny_dfs.append(tiny_df)
                else:
                    continue
            tiny_df = pd.concat(tiny_dfs)
            dil_dfs.append(tiny_df)
        dil_df = pd.concat(dil_dfs)
        final_dfs.append(dil_df)
    output_df = reduce(lambda left, right: pd.merge(left, right, on=['comparison', 'patient_id']),
                       final_dfs)
    return output_df


if __name__ == '__main__':
    main()
