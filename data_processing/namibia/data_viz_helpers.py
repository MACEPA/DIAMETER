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
def get_coef(df, col1='time_point_days', col2='HRP2_pg_ml'):
    regr = linear_model.LinearRegression()
    time = df[col1].values.reshape(-1, 1)
    val = df[col2].values.reshape(-1, 1)
    regr.fit(time, val)
    coef = np.float(regr.coef_)
    return coef


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


def hrp2_outliering(main_data):
    all_dfs = []
    # loop through patients
    for pid in main_data['patient_id'].unique():
        # subset data to just PID of interest
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        # fetch and sort day values
        all_times = pid_data['time_point_days'].unique().tolist()
        all_times.sort()
        # get first three days
        first_days = all_times[:3]
        # get other times
        later_days = all_times[3:]
        # get initial HRP2 value
        initial_pg = pid_data.loc[pid_data['time_point_days'] == first_days[0], 'HRP2_pg_ml'].item()
        # subset a dataframe of just the first three days
        early_df = pid_data.loc[pid_data['time_point_days'].isin(first_days)]
        # get the mean of the first three HRP2 values
        mean_val = early_df['HRP2_pg_ml'].mean()
        # subset a dataframe to all the other days
        other_df = pid_data.loc[~pid_data['time_point_days'].isin(first_days)]
        # start the outliering here
        i = 3
        outlier_vals = []
        while i <= len(later_days) + 1:
            try:
                time_vals = all_times[i - 1:i + 2]
                num_days = 3
            except IndexError:
                time_vals = all_times[i - 1:i]
                num_days = 2
            prev_val = pid_data.loc[pid_data['time_point_days'] == time_vals[0], 'HRP2_pg_ml'].item()
            day_val = pid_data.loc[pid_data['time_point_days'] == time_vals[1], 'HRP2_pg_ml'].item()
            if num_days == 3:
                next_val = pid_data.loc[pid_data['time_point_days'] == time_vals[2], 'HRP2_pg_ml'].item()
                cond_a = ((day_val - prev_val) > 1.5) & ((day_val - next_val) > 1.5)
                cond_b = ((prev_val - day_val) > 1.5) & ((next_val - day_val) > 1.5)
                if cond_a or cond_b:
                    outlier_vals.append(time_vals[1])
            else:
                if abs(day_val - prev_val) > 2:
                    outlier_vals.append(time_vals[1])
            i += 1
        outliered_df = pid_data.loc[~pid_data['time_point_days'].isin(outlier_vals)]
        all_dfs.append(outliered_df)
    return pd.concat(all_dfs)


def hrp2_grouping(main_data):
    all_dfs = []
    # loop through patients
    for pid in main_data['patient_id'].unique():
        # subset data to just PID of interest
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        # fetch and sort day values
        all_times = pid_data['time_point_days'].unique().tolist()
        all_times.sort()
        # get first three days
        first_days = all_times[:3]
        # get other times
        later_days = all_times[3:]
        # get initial HRP2 value
        initial_pg = pid_data.loc[pid_data['time_point_days'] == first_days[0], 'HRP2_pg_ml'].item()
        # subset a dataframe of just the first three days
        early_df = pid_data.loc[pid_data['time_point_days'].isin(first_days)]
        # second and third points must be clearing (green)
        early_df['group'] = 'green'
        # first point must be one of either symptomatic (red) or chronic (yellow)
        first_fever = pid_data['fever48_r'].unique()[0]
        if first_fever == 1:
            early_df.loc[early_df['time_point_days'] == first_days[0], 'group'] = 'red'
        else:
            early_df.loc[early_df['time_point_days'] == first_days[0], 'group'] = 'yellow'
        # get the mean of the first three HRP2 values
        mean_val = early_df['HRP2_pg_ml'].mean()
        # subset a dataframe to all the other days
        other_df = pid_data.loc[~pid_data['time_point_days'].isin(first_days)]
        # set to clearing as a base line
        other_df['group'] = 'green'
        # get the date of retreatment if it exists
        when_retreated = other_df['when_retreated'].unique()[0]
        # start the complex grouping here!
        the_rest = []
        i = 0
        end_val = 4
        max_run = 4
        chronic_vals = []
        longest_section = []
        set_all = False
        if first_fever != 1:
            final_pg = pid_data.loc[pid_data['time_point_days'] == later_days[-1], 'HRP2_pg_ml'].item()
            if (final_pg > 2.9) or ((initial_pg - final_pg) < 1):
                set_all = True
        while (end_val < len(all_times)) & (len(the_rest) != 1):
            time_vals = later_days[i:end_val]
            small_df = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
            coef_a = get_coef(small_df)
            while (coef_a > -.02) & (len(time_vals) != 1) & (end_val < len(all_times)):
                end_val = end_val + 1
                time_vals = later_days[i:end_val]
                bigger_df = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
                coef_b = get_coef(bigger_df)
                if coef_b > -.01:
                    final_4 = time_vals[-4:]
                    check_df = pid_data.loc[pid_data['time_point_days'].isin(final_4)]
                    coef_c = get_coef(check_df)
                    if coef_c > -.05:
                        current_run = len(time_vals)
                        if current_run > max_run:
                            max_run = current_run
                            longest_section = time_vals[:-1]
                            chronic_vals += longest_section
            i = end_val - 1
            try:
                all_times[i + 4]
                end_val = i + 4
            except IndexError:
                the_rest = all_times[i:]
                end_val = i + len(the_rest)
        chronic_vals = list(set(chronic_vals))
        chronic_df = other_df.loc[other_df['time_point_days'].isin(chronic_vals)]
        other_df = other_df.loc[~other_df['time_point_days'].isin(chronic_vals)]
        chronic_df['group'] = 'yellow'
        chronic_df.loc[chronic_df['HRP2_pg_ml'] < 1.5, 'group'] = 'green'
        other_df = pd.concat([other_df, chronic_df])
        combined = pd.concat([early_df, other_df])
        if set_all:
            combined['group'] = 'yellow'
        if when_retreated is not None:
            if pid != 'pa-026':  # patient 26 doesn't make sense and was treated after the study ended
                combined.loc[combined['time_point_days'] == when_retreated, 'group'] = 'red'
                before_retreat = combined.loc[combined['time_point_days'] < when_retreated]
                before_retreat.loc[before_retreat['group'] == 'green', 'group'] = 'yellow'
                before_days = before_retreat['time_point_days'].unique().tolist()
                combined = combined.loc[~combined['time_point_days'].isin(before_days)]
                combined = pd.concat([before_retreat, combined])
        all_dfs.append(combined)
    return pd.concat(all_dfs)


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
