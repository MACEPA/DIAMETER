import numpy as np
import pandas as pd


def parse_id(df):
    all_ids = df['participant_id'].split('-')
    return all_ids[2]


def main(input_dir):
    semi_formatted = pd.read_csv('{}/formatted_4plex_NIH_clinical.csv'.format(input_dir))
    semi_formatted['Study ID number'] = semi_formatted.apply(parse_id, axis=1)
    semi_formatted['timepoint_days'] = semi_formatted['timepoint_days'].apply(int)
    semi_formatted['Study ID number'] = semi_formatted['Study ID number'].apply(int)
    semi_formatted = semi_formatted[['sample_id', 'participant_id', 'timepoint_days', 'RDT_pos', 'HRP2_pg_ml',
                                     'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml', 'HRP2_result',
                                     'LDH_Pan_result', 'LDH_Pv_result', 'Study ID number']]

    fixed_days = []
    for sid in semi_formatted['Study ID number'].unique():
        sid_df = semi_formatted.loc[semi_formatted['Study ID number'] == sid]
        all_days = sid_df['timepoint_days'].unique().tolist()
        fixed_list = [day for day in all_days if ~np.isnan(day)]
        min_day = min(fixed_list)
        sid_df['timepoint_days'] = sid_df['timepoint_days'].subtract(min_day)
        fixed_days.append(sid_df)
    fixed_df = pd.concat(fixed_days)
    fixed_df.sort_values(['Study ID number', 'timepoint_days'], inplace=True)

    dem_par = pd.read_csv('{}/demographic_parasitemia_data.csv'.format(input_dir))
    dem_par = dem_par[['Study ID number', 'P. falciparum', 'P. malariae', 'P. ovale', 'Visit Date',
                       'Age at visit', 'studyday']]
    dem_par.rename(columns={'Visit Date': 'date', 'studyday': 'timepoint_days',
                            'Age at visit': 'age_yrs'}, inplace=True)

    med_info = pd.read_csv('{}/malaria_conmeds.csv'.format(input_dir))
    med_info = med_info.loc[med_info['Drug1 Name'] != 'COMPLEX B']
    med_info = med_info.loc[med_info['Drug1 Indication'].isin(['ACCES PALUSTRE', 'ACCES PALUSTREQ'])]
    med_info = med_info[['Study ID number', 'Drug1 Name', 'Drug1 Start Date']]
    med_info.rename(columns={'Drug1 Name': 'drug', 'Drug1 Start Date': 'date'}, inplace=True)

    check_combo = med_info.merge(dem_par, how='outer', on=['Study ID number', 'date'], suffixes=(False, False))
    check_combo.sort_values('Study ID number', inplace=True)
    check_combo['timepoint_days'] = check_combo['timepoint_days'].apply(lambda x: x if np.isnan(x) else int(x))
    check_combo['Study ID number'] = check_combo['Study ID number'].apply(lambda x: x if np.isnan(x) else int(x))

    all_combo = check_combo.merge(semi_formatted, how='outer',
                                  on=['Study ID number', 'timepoint_days'], suffixes=(False, False))
    all_combo = all_combo[['Study ID number', 'sample_id', 'participant_id', 'date', 'timepoint_days', 'drug',
                           'RDT_pos', 'P. falciparum', 'P. malariae', 'P. ovale', 'age_yrs',
                           'HRP2_pg_ml', 'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml',
                           'HRP2_result', 'LDH_Pan_result', 'LDH_Pv_result']]
    all_combo.sort_values(['Study ID number'], inplace=True)
    all_combo['date'] = pd.to_datetime(all_combo['date'])

    rebuilt_df = []
    for sid in all_combo['Study ID number'].unique():
        sid_df = all_combo.loc[all_combo['Study ID number'] == sid]
        pids = sid_df['participant_id'].unique().tolist()
        pid = [pid for pid in pids if pid is not np.nan]
        if len(pid) > 0:
            sid_df['participant_id'] = pid[-1]
        all_days = sid_df['timepoint_days'].unique().tolist()
        fixed_days = [day for day in all_days if ~np.isnan(day)]
        min_day = min(fixed_days)
        sid_df['timepoint_days'] = sid_df['timepoint_days'].subtract(min_day)
        sid_df['zero_date'] = sid_df.loc[sid_df['timepoint_days'] == 0, 'date'].item()
        sid_df['zero_date'] = pd.to_datetime(sid_df['zero_date'])
        sid_df['date_dif'] = sid_df['date'] - sid_df['zero_date']
        sid_df['date_dif'] = sid_df['date_dif'] / np.timedelta64(1, 'D')
        sid_df.drop(['date', 'zero_date'], axis=1, inplace=True)
        rebuilt_df.append(sid_df)
    rebuilt_df = pd.concat(rebuilt_df)
    rebuilt_df.sort_values(['Study ID number', 'timepoint_days'], inplace=True)
    rebuilt_df.rename(columns={'Study ID number': 'id_number'}, inplace=True)

    rebuilt_df.to_csv('{}/for_viz.csv'.format(input_dir), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        help='Input directory')
    args = parser.parse_args()
    main(input_dir=args.input_dir)
