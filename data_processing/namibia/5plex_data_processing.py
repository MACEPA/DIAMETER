import os
import argparse
import numpy as np
import pandas as pd
from functools import partial, reduce
# import helper functions
from data_processing_helpers import (run_5plex_compare, return_5plex_decisions,
                                     fix_concentrations, build_dil_constants,
                                     set_hrp2_alerts)
# import constants
from data_processing_helpers import THRESHOLDS


# function for determining which dilution value to use
def decider(base_df, base_dil):
    # create the dilution constants via the base dilution
    dil_cons = build_dil_constants(base_dil)
    # create an empty list to fill with small dfs, which will be combined
    analyte_dfs = []
    # create an empty dictionary to fill with errors associated with patient IDs
    error_pids = {}
    # iterate over analytes
    for analyte in THRESHOLDS[5].keys():
        patient_dfs = []
        # iterate over patient_ids
        for pid in base_df['patient_id'].unique():
            patient_data = base_df.loc[base_df['patient_id'] == pid]
            # get number of dilutions
            dilution_values = sorted([val for val in patient_data['concentration'].unique() if val != '1'], key=len)
            # set initial best decision to neat (1)
            best_decision = '1'
            # iterate over dilution values
            for current_dilution in dilution_values:
                # subset to dilutions
                best_dil_data = patient_data.loc[patient_data['concentration'].isin([best_decision])]
                current_dil_data = patient_data.loc[patient_data['concentration'].isin([current_dilution])]
                # create partial function for generating decision vectors
                partial_compare_best = partial(run_5plex_compare, analyte_val=analyte, dil_val=best_decision)
                partial_compare_current = partial(run_5plex_compare, analyte_val=analyte, dil_val=current_dilution)
                # generate decision vectors
                best_dil_data['decision_vector'] = best_dil_data.apply(partial_compare_best, axis=1)
                current_dil_data['decision_vector'] = current_dil_data.apply(partial_compare_current, axis=1)
                # pull decision matrix for given analyte and concentrations
                decisions = return_5plex_decisions(best_decision, current_dilution)
                decision_matrix = decisions[analyte]
                # construct empty dataframe to hold best values
                best_df = pd.DataFrame(columns=['patient_id', 'errors', analyte,
                                                '{}_dilution'.format(analyte),
                                                '{}_well'.format(analyte)])
                # get decision vectors for each possible decision
                vector_best = best_dil_data.loc[best_dil_data['concentration'] == best_decision,
                                                'decision_vector'].item()
                vector_current = current_dil_data.loc[current_dil_data['concentration'] == current_dilution,
                                                      'decision_vector'].item()
                # get actual decision from decision vectors
                decision = decision_matrix[vector_current, vector_best].item()
                # set value, well, and error based on decision
                if decision in [best_decision, current_dilution]:
                    if decision == best_decision:
                        dil_data = best_dil_data
                    elif decision == current_dilution:
                        dil_data = current_dil_data
                    val = dil_data.loc[dil_data['concentration'] == decision,
                                       analyte].item()
                    well = dil_data.loc[dil_data['concentration'] == decision,
                                        'well'].item()
                    error = dil_data.loc[dil_data['concentration'] == decision,
                                         'error'].item()
                elif decision == 'fail':
                    val = 'fail'
                    well = 'fail'
                    error = np.nan
                    # if it's a fail case, add the error to the list of errors
                    # for the specific patient ID
                    try:
                        error_pids[pid] += ', {} failure'.format(analyte)
                    except KeyError:
                        error_pids[pid] = '{} failure'.format(analyte)
                else:
                    raise ValueError("Unexpected decision value: {}".format(decision))
                # preserve the unselected dilutions
                other_dilutions = [val for val in patient_data['concentration'].unique()]
                other_dilutions = [float(val) for val in other_dilutions if val != 'fail']
                # preserve the maximum dilution, selected or unselected
                max_dilution = max(other_dilutions)
                # preserve the selected dilution
                df_decision = decision if decision != 'fail' else np.nan
                # put all preserved/selected values into the empty dataframe
                best_df = best_df.append({'patient_id': pid, 'errors': error, analyte: val,
                                          '{}_dilution'.format(analyte): df_decision,
                                          '{}_well'.format(analyte): well,
                                          '{}_max_dilution'.format(analyte): max_dilution}, ignore_index=True)
                best_decision = decision
                if decision == 'fail':
                    break
            patient_dfs.append(best_df)
        patient_df = pd.concat(patient_dfs)
        # set all error columns to object for combination later
        patient_df['errors'] = patient_df['errors'].astype('object')
        analyte_dfs.append(patient_df)
    # combine all individual analyte dataframes into one dataframe
    decided = reduce(lambda left, right: pd.merge(left, right, on='patient_id'), analyte_dfs)
    # set HRP2 hook effect alerts
    alert_df = set_hrp2_alerts(decided)
    alert_patients = alert_df['patient_id'].tolist()
    # subset to non-alert patients, then...
    decided = decided.loc[~decided['patient_id'].isin(alert_patients)]
    # recombine with correctly flagged alerts
    decided = pd.concat([alert_df, decided])
    # loop through associated error/patient ID pairs
    for pid in error_pids.keys():
        # subset to individual error(s) associated to patient ID
        error = error_pids[pid]
        # subset dataframe to patient ID where error occurs
        pid_df = decided.loc[decided['patient_id'] == pid]
        # combine all the errors into one big error message
        pid_df['errors'] = pid_df['errors'].apply(lambda x: error if np.isnan(x) else x + ' ' + error)
        # if there's actually an error...
        if len(pid_df) > 0:
            # ...replace current dataframe info with the info that contains the error
            decided = decided.loc[decided['patient_id'] != pid]
            decided = decided.append(pid_df)
    return decided


def main(input_dir, base_dil):
    dfs = []
    input_path = '{}/input_data/20190610'.format(input_dir)
    # get all input data, combine into one df
    for fname in os.listdir(input_path):
        plex_data = pd.read_csv('{}/{}'.format(input_path, fname),
                                skiprows=13, names=['patient_id', 'type', 'well', 'error',
                                                    'HRP2_pg_ml', 'LDH_Pan_pg_ml',
                                                    'LDH_Pv_pg_ml', 'LDH_Pf_pg_ml',
                                                    'CRP_ng_ml'])
        # convert all strings to lowercase
        plex_data = plex_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        # fill empty patient_ids from the preceeding patient_id
        plex_data['patient_id'] = plex_data['patient_id'].fillna(method='ffill')
        # drop patient_ids that are still null
        plex_data = plex_data[~plex_data['patient_id'].isnull()]
        dfs.append(plex_data)
    samples_data = pd.concat(dfs)
    # subset data to just what we want
    samples_data = samples_data.loc[~samples_data['patient_id'].str.contains('ctrl')]
    samples_data = samples_data.loc[~samples_data['type'].isnull()]
    samples_data = samples_data.loc[~samples_data['type'].str.contains('replicate')]
    samples_data = samples_data.drop('type', axis=1)
    # break out concentration from patient string
    samples_data['concentration'] = samples_data['patient_id'].apply(lambda x: x.split(' ')[-1])
    samples_data['patient_id'] = samples_data['patient_id'].apply(lambda x: '_'.join(x.split(' ')[:-1]).replace('/',
                                                                                                                '_'))
    # remove concentration values we don't want
    samples_data = samples_data.loc[(samples_data['concentration'].str.contains('neat|{}'.format(base_dil)))]
    # remove rows where "well" is null
    samples_data = samples_data.loc[~samples_data['well'].isnull()]
    # make concentrations more machine/human readable
    samples_data['concentration'] = samples_data.apply(fix_concentrations, axis=1)
    samples_data = samples_data.sort_values(['patient_id', 'concentration'])
    # run decision function
    output_df = decider(samples_data, base_dil)
    # sort values and output to a csv
    output_df.sort_values('patient_id', inplace=True)
    output_df.set_index('patient_id', inplace=True)
    output_df.to_csv('{}/output_data/final_dilutions.csv'.format(input_dir))
    # also output a csv of partially formatted data, for vetting
    partial_format = samples_data.copy(deep=True)
    partial_format.sort_values('patient_id', inplace=True)
    partial_format.set_index('patient_id', inplace=True)
    partial_format.to_csv('{}/output_data/partially_formatted.csv'.format(input_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop/4plex',
                        help='Input directory')
    parser.add_argument('-bd', '--base_dilution', type=int,
                        default=20,
                        help='Base value for going up chain of dilution (1 -> 50 -> 2500 -> etc.)')
    args = parser.parse_args()
    main(input_dir=args.input_dir, base_dil=args.base_dilution)
