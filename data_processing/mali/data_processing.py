import numpy as np
import pandas as pd


# Function for splitting out the study ID number from the participant ID
def split_study_id(df):
    all_ids = df['participant_id'].split('-')
    return all_ids[2]


def main(input_dir):
    # Pull in Mali study data
    semi_formatted = pd.read_csv('{}/formatted_4plex_NIH_clinical.csv'.format(input_dir))
    # Parse out study ID number
    semi_formatted['Study ID number'] = semi_formatted.apply(split_study_id, axis=1)
    semi_formatted['Study ID number'] = semi_formatted['Study ID number'].apply(int)
    # Make sure timepoint values are integers
    semi_formatted['timepoint_days'] = semi_formatted['timepoint_days'].apply(int)
    # Keep only a subset of columns
    semi_formatted = semi_formatted[['sample_id', 'participant_id', 'timepoint_days', 'RDT_pos', 'HRP2_pg_ml',
                                     'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml', 'HRP2_result',
                                     'LDH_Pan_result', 'LDH_Pv_result', 'Study ID number']]
    fixed_days = []
    # Loop over each study ID number
    for sid in semi_formatted['Study ID number'].unique():
        # Subset to just data for the given study ID number
        sid_df = semi_formatted.loc[semi_formatted['Study ID number'] == sid]
        # Pull out non-null timepoint values
        all_days = sid_df['timepoint_days'].unique().tolist()
        fixed_list = [day for day in all_days if ~np.isnan(day)]
        # Find the minimum timepoint value and use it to scale timepoint so min is always 0
        min_day = min(fixed_list)
        sid_df['timepoint_days'] = sid_df['timepoint_days'].subtract(min_day)
        fixed_days.append(sid_df)
    fixed_df = pd.concat(fixed_days)
    # Sort the new, fixed dataframe by study ID number and timepoint
    fixed_df.sort_values(['Study ID number', 'timepoint_days'], inplace=True)
    # Read in demographc and parasitemia data
    dem_par = pd.read_csv('{}/demographic_parasitemia_data.csv'.format(input_dir))
    # Keep only necessary columns
    dem_par = dem_par[['Study ID number', 'P. falciparum', 'P. malariae', 'P. ovale', 'Visit Date',
                       'Age at visit', 'studyday']]
    # Rename columns
    dem_par.rename(columns={'Visit Date': 'date', 'studyday': 'timepoint_days',
                            'Age at visit': 'age_yrs'}, inplace=True)
    # Read in medication data
    med_info = pd.read_csv('{}/malaria_conmeds.csv'.format(input_dir))
    # Subset to only drugs of interest, through 1) name and 2) indication
    med_info = med_info.loc[med_info['Drug1 Name'] != 'COMPLEX B']
    med_info = med_info.loc[med_info['Drug1 Indication'].isin(['ACCES PALUSTRE', 'ACCES PALUSTREQ'])]
    # Keep only necessary columns after subsetting
    med_info = med_info[['Study ID number', 'Drug1 Name', 'Drug1 Start Date']]
    # Rename columns
    med_info.rename(columns={'Drug1 Name': 'drug', 'Drug1 Start Date': 'date'}, inplace=True)
    # Merge medication and demographic/parasitemia dataframes on study ID and date
    combo = med_info.merge(dem_par, how='outer', on=['Study ID number', 'date'], suffixes=(False, False))
    # Sort combined dataframe by study ID
    combo.sort_values('Study ID number', inplace=True)
    # Convert timepoint and study ID to integers if possible
    combo['timepoint_days'] = combo['timepoint_days'].apply(lambda x: x if np.isnan(x) else int(x))
    combo['Study ID number'] = combo['Study ID number'].apply(lambda x: x if np.isnan(x) else int(x))
    # Merge combined dataframe with the Mali study data
    all_combo = combo.merge(semi_formatted, how='outer', on=['Study ID number', 'timepoint_days'],
                            suffixes=(False, False))
    # Keep only necessary columns after merge
    all_combo = all_combo[['Study ID number', 'sample_id', 'participant_id', 'date', 'timepoint_days', 'drug',
                           'RDT_pos', 'P. falciparum', 'P. malariae', 'P. ovale', 'age_yrs',
                           'HRP2_pg_ml', 'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml',
                           'HRP2_result', 'LDH_Pan_result', 'LDH_Pv_result']]
    # Sort merged dataframe by study ID
    all_combo.sort_values(['Study ID number'], inplace=True)
    # Convert date column to Pandas datetime
    all_combo['date'] = pd.to_datetime(all_combo['date'])
    # Generate the date_dif column, which is used to create visualizations that are comparable across patient IDs
    rebuilt_df = []
    # Loop over each study ID
    for sid in all_combo['Study ID number'].unique():
        # Subset to just data for a given study ID
        sid_df = all_combo.loc[all_combo['Study ID number'] == sid]
        # Get a list of unique, non-null patient IDs assoicated with the given study ID
        pids = sid_df['participant_id'].unique().tolist()
        pid = [pid for pid in pids if pid is not np.nan]
        # If there are multiple patient IDs, keep only the last patient ID and set all patient IDs in the subset
        # data to that patient ID
        if len(pid) > 0:
            sid_df['participant_id'] = pid[-1]
        # Get a list of all unique, non-null timepoints associated with the given study ID
        all_days = sid_df['timepoint_days'].unique().tolist()
        fixed_days = [day for day in all_days if ~np.isnan(day)]
        # Get the minimum of the non-null timepoints
        min_day = min(fixed_days)
        # Make the timepoints 0 indexed, so the minimum value is always timepoint 0
        sid_df['timepoint_days'] = sid_df['timepoint_days'].subtract(min_day)
        # Create a 'zero_date' column that contains the date at timepoint 0
        sid_df['zero_date'] = sid_df.loc[sid_df['timepoint_days'] == 0, 'date'].item()
        sid_df['zero_date'] = pd.to_datetime(sid_df['zero_date'])
        # Create a "date_dif' column that contains the different between the zero date and the date for any given row
        sid_df['date_dif'] = sid_df['date'] - sid_df['zero_date']
        # Convert the date_dif to a usable format
        sid_df['date_dif'] = sid_df['date_dif'] / np.timedelta64(1, 'D')
        # Get rid of the 'date' and 'zero_date' columns, keeping only date_dif
        sid_df.drop(['date', 'zero_date'], axis=1, inplace=True)
        rebuilt_df.append(sid_df)
    rebuilt_df = pd.concat(rebuilt_df)
    # Sort the final dataframe based on study ID and timepoint
    rebuilt_df.sort_values(['Study ID number', 'timepoint_days'], inplace=True)
    # Rename the study ID column to id_number
    rebuilt_df.rename(columns={'Study ID number': 'id_number'}, inplace=True)
    # Save the final dataframe as a CSV
    rebuilt_df.to_csv('{}/for_viz.csv'.format(input_dir), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        help='Input directory')
    args = parser.parse_args()
    main(input_dir=args.input_dir)
