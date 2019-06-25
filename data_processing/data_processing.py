import os
import pandas as pd
import numpy as np
from functools import partial, reduce
# import helper function
from data_processing.data_processing_helpers import (run_compare, return_decisions,
                                                     fix_concentrations, split_time,
                                                     remove_time)
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
        (samples_data['concentration'].str.contains('neat|50'))]
    samples_data = samples_data.loc[~samples_data['concentration'].str.contains('low volume')]
    samples_data = samples_data.loc[~samples_data['well'].isnull()]
    # fix concentrations
    samples_data['concentration'] = samples_data.apply(fix_concentrations, axis=1)
    samples_data = samples_data.sort_values(['patient_id', 'well'])

    # subset the data to just duplicates
    duplicates = samples_data.loc[samples_data.duplicated(subset=['patient_id', 'concentration'], keep=False)]
    deduped_dfs = []
    for analyte in THRESHOLDS.keys():
        dup_analyte = duplicates[['patient_id', 'well', 'error', 'concentration', analyte]]
        pid_dfs = []
        for pid in duplicates['patient_id'].unique():
            dup_data = dup_analyte.loc[dup_analyte['patient_id'] == pid]
            con_dfs = []
            for concentration in dup_data['concentration'].unique():
                fill_df = pd.DataFrame(columns=['patient_id', 'well', 'error',
                                                'concentration', analyte])
                # everything until this is subsetting to specific data
                dup_con = dup_data.loc[dup_data['concentration'] == concentration]
                # get the values for the duplicate concentrations
                values = dup_con[analyte]
                # also preserve wells and errors for duplicate concentrations
                wells = dup_con['well'].tolist()
                wells = ''.join(c for c in str(wells) if c not in ["[", "]", "'"])
                errors = dup_con['error'].tolist()
                non_nan_error = [e for e in errors if e is not np.nan]
                if not non_nan_error:
                    errors = np.nan
                else:
                    errors = non_nan_error
                try:
                    # if they're both real numbers, take the average
                    values = [float(val) for val in values.tolist()]
                    val = sum(values) / len(values)
                except ValueError:
                    values = values.tolist()
                    num_vals = [val for val in values if ('<' not in val) & ('>' not in val)]
                    # if one is a real number, take that one
                    if len(num_vals) == 1:
                        val = num_vals[0]
                    # if both are non-real, we assume they're the same. maybe sketchy?
                    else:
                        val = values[0]
                fill_df = fill_df.append({'patient_id': pid, 'well': wells, 'error': errors,
                                          'concentration': concentration, analyte: val}, ignore_index=True)
                con_dfs.append(fill_df)
            con_df = pd.concat(con_dfs)
            pid_dfs.append(con_df)
        pid_df = pd.concat(pid_dfs)
        deduped_dfs.append(pid_df)
    deduped = reduce(lambda left, right: pd.merge(left, right, on=['patient_id', 'well', 'error', 'concentration']),
                     deduped_dfs)
    # replace old duplicated values with new dedeuplicated values
    no_duplicates = samples_data.drop_duplicates(subset=['patient_id', 'concentration'], keep=False)
    no_duplicates = pd.concat([no_duplicates, deduped])

    # create an empty list to fill with small dfs, which will be combined
    analyte_dfs = []
    # run counts for decision on what to keep
    for analyte in THRESHOLDS.keys():
        patient_dfs = []
        # iterate over patient_ids
        for i in samples_data['patient_id'].unique():
            patient_data = no_duplicates.loc[no_duplicates['patient_id'] == i]
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
                best_df = pd.DataFrame(columns=['patient_id', 'error', analyte,
                                                '{}_dilution'.format(analyte),
                                                '{}_well'.format(analyte)])
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
                    error = dil_data.loc[dil_data['concentration'] == decision,
                                         'error'].item()
                elif decision == 'fail':
                    val = 'fail'
                    well = 'fail'
                    error = np.nan
                else:
                    raise ValueError("Unexpected decision value: {}".format(decision))
                other_dilutions = [val for val in patient_data['concentration'].unique()]
                other_dilutions = [float(val) for val in other_dilutions if val != 'fail']
                max_dilution = max(other_dilutions)
                df_decision = decision if decision != 'fail' else np.nan
                best_df = best_df.append({'patient_id': i, 'error': error, analyte: val,
                                          '{}_dilution'.format(analyte): df_decision,
                                          '{}_well'.format(analyte): well,
                                          '{}_max_dilution'.format(analyte): max_dilution}, ignore_index=True)
                best_decision = decision
                if decision == 'fail':
                    break
            patient_dfs.append(best_df)
        patient_df = pd.concat(patient_dfs)
        analyte_dfs.append(patient_df)
    output_df = reduce(lambda left, right: pd.merge(left, right, on='patient_id'), analyte_dfs)
    # split time associated with patient_id into its own column
    output_df['time_point_days'] = output_df.apply(split_time, axis=1)
    output_df['patient_id'] = output_df.apply(remove_time, axis=1)
    output_df.sort_values(['patient_id', 'time_point_days'], inplace=True)
    output_df.set_index(['patient_id', 'time_point_days'], inplace=True)
    output_df.to_csv('C:/Users/lzoeckler/Desktop/4plex/output_data/final_dilutions_time.csv')
    return output_df


if __name__ == '__main__':
    main()
