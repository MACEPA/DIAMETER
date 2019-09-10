import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn import linear_model
from sklearn.metrics import r2_score


# function for removing > and < from strings so they can be cast to float
def clean_strings(val):
    clean = val.replace('<', '')
    clean = clean.replace('>', '')
    if clean == 'fail':
        return np.nan
    return clean


# function for running a linear regression on two columns
# returns the coefficient of the regression and the R2 score
def get_coef(df, col1, col2):
    regr = linear_model.LinearRegression()
    time = df[col1].values.reshape(-1, 1)
    val = df[col2].values.reshape(-1, 1)
    regr.fit(time, val)
    coef = np.float(regr.coef_)
    pred = regr.predict(time)
    score = r2_score(val, pred)
    return coef, score


def rebuild_data(main_data, val_cols):
    # create a dataframe with no short timeseries (<4 points) and
    # no timeseries where HRP2 < 10 initially
    rebuilt_data = []
    for pid in main_data['patient_id'].unique():
        sub_data = main_data.loc[main_data['patient_id'] == pid]
        if len(sub_data) < 4:
            continue
        all_times = sub_data['time_point_days'].unique().tolist()
        start_val = sub_data.loc[sub_data['time_point_days'] == min(all_times), 'HRP2_pg_ml'].item()
        if start_val < 10:
            continue
        rebuilt_data.append(sub_data)
    rebuilt_data = pd.concat(rebuilt_data)
    # return the log10 of all data columns, instead of normal space
    rebuilt_data[val_cols] = rebuilt_data[val_cols].applymap(np.log10)
    return rebuilt_data


def hrp2_complex_grouping(main_data):
    # run HRP2 grouping
    good_df = []
    bad_df = []
    for pid in main_data['patient_id'].unique():
        # subset data to individual patient_id, only HRP2 data
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        all_times = pid_data['time_point_days'].unique().tolist()
        all_times.sort()
        max_run = 3
        i = 0
        end_val = 4
        baddest_section = []
        the_rest_set = set([0])
        while (end_val <= len(all_times)) & (len(the_rest_set) != 0):
            next_start = None
            time_vals = all_times[i:end_val]
            coef_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
            avg_val = coef_data['HRP2_pg_ml'].mean()
            coef, score = get_coef(coef_data, 'time_point_days', 'HRP2_pg_ml')
            extended_time = all_times[end_val:end_val + 4]
        if len(extended_time) > 2:
            extended_data = pid_data.loc[pid_data['time_point_days'].isin(extended_time)]
            extended_coef, extended_score = get_coef(extended_data, 'time_point_days', 'HRP2_pg_ml')
        else:
            extended_score = 0
        while (coef > -.03) & (len(time_vals) != 1) & (avg_val > 2.5) & (end_val < len(all_times)):
            end_val = end_val + 1
            time_vals = all_times[i:end_val]
            coef_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
            coef, score = get_coef(coef_data, 'time_point_days', 'HRP2_pg_ml')
            avg_val = coef_data['HRP2_pg_ml'].mean()
            end_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals[-4:])]
            end_coef, end_score = get_coef(end_data, 'time_point_days', 'HRP2_pg_ml')
            extended_time = all_times[end_val:end_val + 4]
            condition1 = (coef > -.03) & (avg_val > 2.5) & (score < .3) & (end_score < .4)
            condition2 = (coef > 0) & (avg_val > 2.5) & (score < .3)
            if condition1 or condition2:
                current_run = len(time_vals)
                next_start = end_val - 1
                if current_run > max_run:
                    max_run = current_run
                    baddest_section = time_vals
        if next_start:
            i = next_start
        else:
            i = end_val - 3
        try:
            all_times[i + 4]
            end_val = i + 4
        except IndexError:
            the_rest = all_times[i:]
            the_rest_set = set(the_rest) - set(time_vals)
            end_val = i + len(the_rest)
        good_vals = pid_data.loc[~pid_data['time_point_days'].isin(baddest_section)]
        good_df.append(good_vals)
        bad_vals = pid_data.loc[pid_data['time_point_days'].isin(baddest_section)]
        bad_df.append(bad_vals)
    good_df = pd.concat(good_df)
    bad_df = pd.concat(bad_df)
    good_df['group'] = 'blue'
    bad_df['group'] = 'red'
    combo_df = pd.concat([good_df, bad_df])
    return combo_df


def hrp2_ratio_grouping(main_data):
    good_df = []
    bad_df = []
    for pid in main_data['patient_id'].unique():
        bad_days = []
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        pid_data.sort_values('time_point_days', inplace=True)
        all_times = pid_data['time_point_days'].unique().tolist()
        all_times.sort()
        for day in all_times:
            day_df = pid_data.loc[pid_data['time_point_days'] == day]
            day_df['ratio'] = day_df['LDH_Pan_pg_ml'].divide(day_df['HRP2_pg_ml'])
            if (day_df['ratio'].item() > .8) & (day_df['HRP2_pg_ml'].item() > 4):
                bad_days.append(day)
        good_vals = pid_data.loc[~pid_data['time_point_days'].isin(bad_days)]
        good_df.append(good_vals)
        bad_vals = pid_data.loc[pid_data['time_point_days'].isin(bad_days)]
        bad_df.append(bad_vals)
    good_df = pd.concat(good_df)
    bad_df = pd.concat(bad_df)
    good_df['group'] = 'blue'
    good_df['ratio'] = good_df['LDH_Pan_pg_ml'].divide(good_df['HRP2_pg_ml'])
    bad_df['group'] = 'red'
    bad_df['ratio'] = bad_df['LDH_Pan_pg_ml'].divide(bad_df['HRP2_pg_ml'])
    combo_df = pd.concat([good_df, bad_df])
    combo_df['returned_with_fever'].fillna('No', inplace=True)
    combo_df['retreated'] = combo_df['retreated'].apply(lambda x: 'No' if x == 0.0 else x)
    combo_df['retreated'] = combo_df['retreated'].apply(lambda x: 'Yes' if x == 1.0 else x)
    return combo_df


# get all colors and shapes for association
all_colors = cm.rainbow(np.linspace(0, 1, 8))
all_dilutions = ['1', '50', '2500', '125000', '6250000', '312500000', '15625000000',
                 '781250000000']
all_shapes = ['+', 'v', 's', 'p', 'd', '^', '.', '*']

# associate colors to different dilution values
combo = zip(all_dilutions, all_colors)
COLOR_DICT = {dil: val for dil, val in combo}
COLOR_DICT['fail'] = np.array([0.0, 0.0, 0.0, 0.0])

# associate shapes to different dilution values
shape_combo = zip(all_dilutions, all_shapes)
SHAPE_DICT = {dil: val for dil, val in shape_combo}

# list of all analytes and assoicated name components
ANALYTE_INFO = {'HRP2_pg_ml': ('HRP2', 'pg/ml'), 'LDH_Pan_pg_ml': ('LDH_Pan', 'pg/ml'),
                'LDH_Pv_pg_ml': ('LDH_Pv', 'pg/ml'), 'CRP_ng_ml': ('CRP', 'ng/ml')}
