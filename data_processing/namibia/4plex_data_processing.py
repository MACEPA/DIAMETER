import os
import argparse
import numpy as np
import pandas as pd
from functools import partial, reduce
# import helper functions
from data_processing_helpers import (run_compare, return_decisions,
                                     fix_concentrations, split_time,
                                     remove_time, remove_day, read_4plex,
                                     read_5plex)
# import constants
from data_processing_helpers import THRESHOLDS_4PLEX, THRESHOLDS_5PLEX


# function for combining duplicates
def deduplicate(duplicate_df, plex):
    # create an empty list to fill with small dfs, which will be combined
    deduped_dfs = []
    # iterate over analytes
    if plex == 4:
        thresholds = THRESHOLDS_4PLEX
    elif plex == 5:
        thresholds = THRESHOLDS_5PLEX
    for analyte in thresholds.keys():
        # subset to columns of interest
        dup_analyte = duplicate_df[['patient_id', 'well', 'error', 'concentration', analyte]]
        pid_dfs = []
        # iterate over patient_ids
        for pid in duplicate_df['patient_id'].unique():
            # subset to specific patient_id
            dup_data = dup_analyte.loc[dup_analyte['patient_id'] == pid]
            con_dfs = []
            # iterate over duplicate concentrations
            for concentration in dup_data['concentration'].unique():
                # create an empty dataframe to fill
                fill_df = pd.DataFrame(columns=['patient_id', 'well', 'error',
                                                'concentration', analyte])
                # subset to specific concentration value
                dup_con = dup_data.loc[dup_data['concentration'] == concentration]
                # get the values for the duplicate concentrations
                values = dup_con[analyte]
                # also preserve wells and errors for duplicate concentrations
                wells = dup_con['well'].tolist()
                wells = ''.join(c for c in str(wells) if c not in ["[", "]", "'"])
                errors = dup_con['error'].tolist()
                non_nan_error = [e for e in errors if e is not np.nan]
                if non_nan_error:
                    errors = non_nan_error
                else:
                    errors = np.nan
                try:
                    # if they're both real numbers, take the average
                    values = [float(val) for val in values.tolist()]
                    val = sum(values) / len(values)
                except ValueError:
                    # otherwise...
                    values = values.tolist()
                    num_vals = [val for val in values if ('<' not in val) & ('>' not in val)]
                    # if one is a real number, take that one
                    if len(num_vals) == 1:
                        val = num_vals[0]
                    # if both are non-real, we assume they're the same. maybe sketchy?
                    else:
                        val = values[0]
                # add values to empty dataframe
                fill_df = fill_df.append({'patient_id': pid, 'well': wells, 'error': errors,
                                          'concentration': concentration, analyte: val}, ignore_index=True)
                con_dfs.append(fill_df)
            con_df = pd.concat(con_dfs)
            pid_dfs.append(con_df)
        pid_df = pd.concat(pid_dfs)
        deduped_dfs.append(pid_df)
    deduped = reduce(lambda left, right: pd.merge(left, right, on=['patient_id', 'well', 'error', 'concentration']),
                     deduped_dfs)
    return deduped


# function for determining which dilution value to use
def decider(base_df, plex, base_dil):
    # create an empty list to fill with small dfs, which will be combined
    analyte_dfs = []
    # iterate over analytes
    if plex == 4:
        thresholds = THRESHOLDS_4PLEX
    elif plex == 5:
        thresholds = THRESHOLDS_5PLEX
    for analyte in thresholds.keys():
        patient_dfs = []
        # iterate over patient_ids
        for pid in base_df['patient_id'].unique():
            patient_data = base_df.loc[base_df['patient_id'] == pid]
            # get number of dilutions
            dilution_values = sorted([val for val in patient_data['concentration'].unique() if val != '1'], key=len)
            # set initial best decision to neat (1)
            best_decision = '1'
            # iterate over dilution values
            for max_dilution in dilution_values:
                # subset to dilutions
                dil_data = patient_data.loc[patient_data['concentration'].isin([best_decision, max_dilution])]
                # create partial function for generating decision vectors
                partial_compare = partial(run_compare, analyte_val=analyte, dil_val=max_dilution, base=base_dil)
                # generate decision vectors
                dil_data['decision_vector'] = dil_data.apply(partial_compare, axis=1)
                # pull decision matrix for given analyte and concentrations
                decisions = return_decisions(best_decision, max_dilution)
                decision_matrix = decisions[analyte]
                # construct empty dataframe to hold best values
                best_df = pd.DataFrame(columns=['patient_id', 'errors', analyte,
                                                '{}_dilution'.format(analyte),
                                                '{}_well'.format(analyte)])
                # get decision vectors for each possible decision
                vector_low = dil_data.loc[dil_data['concentration'] == best_decision,
                                          'decision_vector'].item()
                vector_high = dil_data.loc[dil_data['concentration'] == max_dilution,
                                           'decision_vector'].item()
                # get actual decision from decision vectors
                decision = decision_matrix[vector_high, vector_low].item()
                # set value, well, and error based on decision
                if decision in [best_decision, max_dilution]:
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
    decided = reduce(lambda left, right: pd.merge(left, right, on='patient_id'), analyte_dfs)
    return decided


def main(input_dir, plex, base_dil):
    dfs = []
    input_path = '{}/input_data/20190610'.format(input_dir)
    # get all input data, combine into one df
    for fname in os.listdir(input_path):
        if plex == 4:
            plex_data = read_4plex(input_path, fname)
        elif plex == 5:
            plex_data = read_5plex(input_path, fname)
        else:
            raise ValueError("Unexpected plex value: {}".format(plex))
        # convert all strings to lowercase
        plex_data = plex_data.applymap(lambda x: x.lower() if isinstance(x, str) else x)
        # fill empty patient_ids from the preceeding patient_id
        plex_data['patient_id'] = plex_data['patient_id'].fillna(method='ffill')
        # drop patient_ids that are still null
        plex_data = plex_data[~plex_data['patient_id'].isnull()]
        dfs.append(plex_data)
    samples_data = pd.concat(dfs)
    # subset data to just what we want
    samples_data = samples_data.loc[~samples_data['type'].isnull()]
    if plex == 4:
        samples_data = samples_data.loc[~samples_data['type'].str.contains('pixel')]
        samples_data = samples_data.loc[samples_data['patient_id'].str.contains('pa-')]
    elif plex == 5:
        samples_data = samples_data.loc[~samples_data['patient_id'].str.contains('ctrl')]
        samples_data = samples_data.loc[~samples_data['type'].str.contains('replicate')]
        samples_data = samples_data.loc[~samples_data['type'].isnull()]
    samples_data = samples_data.drop('type', axis=1)
    # break out concentration from patient string
    samples_data['concentration'] = samples_data['patient_id'].apply(lambda x: x.split(' ')[-1])
    if plex == 4:
        samples_data['patient_id'] = samples_data['patient_id'].apply(lambda x: x.partition(' ')[0])
    elif plex == 5:
        samples_data['patient_id'] = samples_data['patient_id'].apply(lambda x: '_'.join(x.split(' ')[:3]).replace('/',
                                                                                                                   '_'))
    # remove concentration values we don't want
    samples_data = samples_data.loc[(samples_data['concentration'].str.contains('neat|{}'.format(base_dil)))]
    samples_data = samples_data.loc[~samples_data['concentration'].str.contains('low volume')]
    # remove rows where "well" is null
    samples_data = samples_data.loc[~samples_data['well'].isnull()]
    # make concentrations more machine/human readable
    samples_data['concentration'] = samples_data.apply(fix_concentrations, axis=1)
    samples_data = samples_data.sort_values(['patient_id', 'concentration'])
    # subset the data to just duplicates
    duplicates = samples_data.loc[samples_data.duplicated(subset=['patient_id', 'concentration'], keep=False)]
    # run deduplicating function, return deduplicated df
    deduped = deduplicate(duplicates)
    # replace old duplicated values with new dedeuplicated values
    no_duplicates = samples_data.drop_duplicates(subset=['patient_id', 'concentration'], keep=False)
    no_duplicates = pd.concat([no_duplicates, deduped])
    # run decision function
    output_df = decider(no_duplicates)
    # split time associated with patient_id into its own column
    output_df['time_point_days'] = output_df.apply(split_time, axis=1, plex=plex)
    output_df['patient_id'] = output_df.apply(remove_time, axis=1, plex=plex)
    # sort values and output to a csv
    output_df.sort_values(['patient_id', 'time_point_days'], inplace=True)
    output_df.set_index(['patient_id', 'time_point_days'], inplace=True)
    output_df.to_csv('{}/output_data/final_dilutions.csv'.format(input_dir))
    # also output a csv of partially formatted data, for vetting
    partial_format = samples_data.copy(deep=True)
    partial_format['time_point_days'] = partial_format.apply(split_time, axis=1, plex=plex)
    partial_format['patient_id'] = partial_format.apply(remove_time, axis=1, plex=plex)
    partial_format.sort_values(['patient_id', 'time_point_days'], inplace=True)
    partial_format.set_index(['patient_id', 'time_point_days'], inplace=True)
    partial_format.to_csv('{}/output_data/partially_formatted.csv'.format(input_dir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop/4plex',
                        help='Input directory')
    parser.add_argument('-p', '--plex', type=int,
                        default=5,
                        help="4plex vs 5plex (or any future nplex)")
    parser.add_argument('-bd', '--base_dil', type=int,
                        default=50,
                        help='Base dilution value beyond neat (1)')
    args = parser.parse_args()
    main(input_dir=args.input_dir, plex=args.plex, base_dil=args.base_dil)
