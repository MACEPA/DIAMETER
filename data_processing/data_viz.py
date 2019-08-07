import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
from data_processing.data_viz_helpers import clean_strings
from data_processing.data_viz_helpers import (COLOR_DICT, SHAPE_DICT,
                                              ANALYTE_INFO)


def analyte_shapes(main_data, analyte, analyte_name):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    pp = PdfPages('{}/{}_graphs.pdf'.format(output_fp, analyte))
    # create individual graphs for each patient_id
    for pid in main_data['patient_id'].unique():
        # subset data
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        plot_data = pid_data[['patient_id', 'time_point_days', analyte,
                              '{}_dilution'.format(analyte),
                              '{}_max_dilution'.format(analyte)]]
        plot_data[analyte] = plot_data[analyte].apply(clean_strings)
        plot_data[analyte] = plot_data[analyte].apply(float)
        plot_data = plot_data.loc[~plot_data[analyte].isnull()]
        vals = plot_data[analyte].tolist()
        # try to captures "interesting" graphs with a red color
        try:
            if (vals[0] < vals[1]) | (vals[0] < vals[2]):
                plt_color = 'red'
            else:
                plt_color = 'blue'
        except IndexError:
            plt_color = 'blue'
        f = plt.figure()
        f.add_subplot()
        # line plot with "interest" color
        plt.plot(plot_data['time_point_days'], plot_data[analyte], color=plt_color, alpha=0.3)
        dil_vals = plot_data['{}_dilution'.format(analyte)].tolist()
        dil_vals = [val for val in dil_vals if not np.isnan(val)]
        dil_vals = [int(val) if not np.isnan(val) else val for val in dil_vals]
        dil_vals = [str(val) if val != 'fail' else val for val in dil_vals]
        vals = plot_data[analyte].tolist()
        time = plot_data['time_point_days'].tolist()
        # return the maximum dilution available for each data point
        maximum = plot_data['{}_max_dilution'.format(analyte)]
        data = zip(time, vals)
        # plot the data points in a scatter with color indicating dilution
        # and size indicating maximum dilution
        color_legend = []
        used_dil = []
        shape_legend = []
        used_shapes = []
        for data, group, max_dil in zip(data, dil_vals, maximum):
            x, y = data
            color = COLOR_DICT[group]
            max_dil = str(int(max_dil))
            shape = SHAPE_DICT[max_dil]
            plt.scatter(x, y, c=[color], marker=shape, s=120, alpha=1.0)
            if group not in used_dil:
                color_legend.append(Line2D([0], [0], marker='o', color='w', label=group,
                                           markerfacecolor=color, markersize=15))
                used_dil.append(group)
            if shape not in used_shapes:
                shape_legend.append(Line2D([0], [0], marker=shape, color='w', label=max_dil,
                                           markerfacecolor='k', markersize=15))
                used_shapes.append(shape)
        # plot in log scale
        plt.yscale('log')
        true_analyte = analyte_name[0]
        title = "analyte: {}, patient_id: {}".format(true_analyte, pid)
        plt.title(title)
        # add the two different legends in, for color and size
        first_legend = plt.legend(handles=color_legend, loc='best',
                                  title='Dilution used')
        second_legend = plt.legend(handles=shape_legend,
                                   bbox_to_anchor=(1.00, 0.5),
                                   title='Max dilution')
        plt.gca().add_artist(first_legend)
        plt.tight_layout()
        pp.savefig(f, bbox_extra_artists=[first_legend, second_legend], bbox_inches='tight')
        plt.close()
    pp.close()


def analyte_point_individuals(main_data, analyte, analyte_name):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    pp = PdfPages('{}/{}_all_individuals.pdf'.format(output_fp, analyte))
    # subset data to analyte of interest
    point_data = main_data[['patient_id', 'time_point_days', analyte,
                            '{}_dilution'.format(analyte),
                            '{}_max_dilution'.format(analyte)]]
    # clean strings to remove '>'/'<', convert 'fail' to NaN
    point_data[analyte] = point_data[analyte].apply(clean_strings)
    # convert strings to float
    point_data[analyte] = point_data[analyte].apply(float)
    # take log of the floats
    point_data[analyte] = point_data[analyte].apply(math.log)
    # only keep non-null values
    point_data = point_data.loc[~point_data[analyte].isnull()]
    # run a simple linear regression on the logged data
    regr = linear_model.LinearRegression()
    time = point_data['time_point_days'].values.reshape(-1, 1)
    val = point_data[analyte].values.reshape(-1, 1)
    regr.fit(time, val)
    pred = regr.predict(time)
    # return the R2 value, the slope, and the interecpt of the model
    score = r2_score(val, pred)
    coef = np.float(regr.coef_)
    intercept = np.float(regr.intercept_)
    # plot the points as well as the linear regression
    f = plt.figure()
    plt.scatter(time, val)
    plt.plot(time, pred, color='green', label="Line of best fit")
    # label the plot and the axes
    true_analyte = analyte_name[0]
    units = analyte_name[1]
    title = """Analyte: {}, Slope: {}, \nIntercept: {}, R2: {}""".format(
        true_analyte, round(coef, 8), round(intercept, 8), score)
    plt.title(title)
    plt.xlabel("Timepoint, in days")
    plt.ylabel("Log of analyte value, {}".format(units))
    plt.legend()
    plt.tight_layout()
    # save the plot
    pp.savefig(f)
    plt.close()
    pp.close()


def analyte_connected_individuals(main_data, analyte, analyte_name):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    pp = PdfPages('{}/{}_all_connected.pdf'.format(output_fp, analyte))
    # create main graph
    f = plt.figure()
    f.add_subplot()
    for pid in main_data['patient_id'].unique():
        # plot each patient ID separately, on the same main graph
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        plot_data = pid_data[['patient_id', 'time_point_days', analyte,
                              '{}_dilution'.format(analyte),
                              '{}_max_dilution'.format(analyte)]]
        # clean strings to remove '>'/'<', convert 'fail' to NaN
        plot_data[analyte] = plot_data[analyte].apply(clean_strings)
        # convert strings to float
        plot_data[analyte] = plot_data[analyte].apply(float)
        # take log of the floats
        plot_data[analyte] = plot_data[analyte].apply(math.log)
        # only keep non-null values
        plot_data = plot_data.loc[~plot_data[analyte].isnull()]
        # plot the individual
        plt.plot(plot_data['time_point_days'], plot_data[analyte])
    # label the plot and the axes
    true_analyte = analyte_name[0]
    units = analyte_name[1]
    plt.title('Individuals over time\nAnalyte: {}'.format(true_analyte))
    plt.xlabel("Timepoint, in days")
    plt.ylabel("Log of analyte value, {}".format(units))
    plt.tight_layout()
    # save the plot
    pp.savefig(f)
    plt.close()
    pp.close()


def get_coef(df):
    regr = linear_model.LinearRegression()
    time = df['time_point_days'].values.reshape(-1,1)
    val = df['HRP2_pg_ml'].values.reshape(-1,1)
    regr.fit(time, val)
    coef = np.float(regr.coef_)
    pred = regr.predict(time)
    score = r2_score(val, pred)
    return coef, score


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
        the_rest = []
        i = 0
        end_val = 4
        baddest_section = []
        the_rest_set = set([0])
        while (end_val <= len(all_times)) & (len(the_rest_set) != 0):
            next_start = None
            time_vals = all_times[i:end_val]
            coef_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
            avg_val = coef_data['HRP2_pg_ml'].mean()
            coef, score = get_coef(coef_data)
            extended_time = all_times[end_val:end_val + 4]
        if len(extended_time) > 2:
            extended_data = pid_data.loc[pid_data['time_point_days'].isin(extended_time)]
            extended_coef, extended_score = get_coef(extended_data)
        else:
            extended_score = 0
        while (coef > -.03) & (len(time_vals) != 1) & (avg_val > 2.5) & (end_val < len(all_times)):
            end_val = end_val + 1
            time_vals = all_times[i:end_val]
            coef_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
            coef, score = get_coef(coef_data)
            avg_val = coef_data['HRP2_pg_ml'].mean()
            end_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals[-4:])]
            end_coef, end_score = get_coef(end_data)
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
    return combo_df


def plot_hrp2_groups(main_data, version):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    pp = PdfPages('{}/HRP2_{}_groups.pdf'.format(output_fp, version))
    for pid in main_data['patient_id'].unique():
        # subset data to individual patient_id, only HRP2 data
        combo = main_data.loc[main_data['patient_id'] == pid]
        combo = combo.sort_values('time_point_days')
        f = plt.figure()
        f.add_subplot()
        title = "patient_id: {}".format(pid)
        plt.plot(combo['time_point_days'], combo['HRP2_pg_ml'],
                 c='black', alpha=0.6)
        plt.plot(combo['time_point_days'], combo['LDH_Pan_pg_ml'], c='green', alpha=0.6)
        plt.scatter(combo['time_point_days'], combo['HRP2_pg_ml'],
                    c=combo['group'])
        plt.title(title)
        plt.ylim(0, 8)
        plt.xlabel('Time point, in days')
        plt.ylabel('Log of HRP2 pg/ml')
        plt.tight_layout()
        pp.savefig(f)
        plt.close()
    pp.close()


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
    rebuilt_data[val_cols] = rebuilt_data[val_cols].applymap(np.log10)
    return rebuilt_data


def main(run_shapes, run_points, run_connected, run_complex_hrp2, run_ratio_hrp2):
    # read in formatted dilution CSV
    input_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    main_data = pd.read_csv('{}/final_dilutions.csv'.format(input_fp))
    # loop through analytes to create different PDFs for each
    for analyte in ANALYTE_INFO.keys():
        analyte_name = ANALYTE_INFO[analyte]
        # produce analyte shape graphs
        if run_shapes:
            analyte_shapes(main_data, analyte, analyte_name)
        # produce analyte individual graphs with trend lines, unconnected
        if run_points:
            analyte_point_individuals(main_data, analyte, analyte_name)
        # produce analyte individual graphs without trend lines, connected
        if run_connected:
            analyte_connected_individuals(main_data, analyte, analyte_name)
        # produce HRP2 groups
    if run_complex_hrp2 or run_ratio_hrp2:
        val_cols = ['HRP2_pg_ml', 'LDH_Pan_pg_ml', 'CRP_ng_ml']
        rebuilt_data = rebuild_data(main_data, val_cols)
        if run_complex_hrp2:
            grouped_data = hrp2_complex_grouping(rebuilt_data)
            plot_hrp2_groups(grouped_data)
        if run_ratio_hrp2:
            grouped_data = hrp2_ratio_grouping(rebuilt_data)
            plot_hrp2_groups(grouped_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-rs', '--run_shapes', action='store_true',
                        help='Whether or not to produce shape graphs')
    parser.add_argument('-rp', '--run_points', action='store_true',
                        help='Whether or not to produce point graphs')
    parser.add_argument('-rc', '--run_connected', action='store_true',
                        help='Whether or not to produce connected graphs')
    parser.add_argument('-rh', '--run_hrp2', action='store_true',
                        help='Whether or not to produce HRP2 grouping graphs')
    args = parser.parse_args()
    main(run_shapes=args.rs, run_points=args.rp, run_connected=args.rc, run_hrp2=args.rh)
