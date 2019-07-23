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


def hrp2_grouping(main_data, analyte, analyte_name):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    pp = PdfPages('{}/{}_manual_groups.pdf'.format(output_fp, analyte))
    for pid in main_data['patient_id'].unique():
        # subset data to individual patient_id, only HRP2 data
        pid_data = main_data.loc[main_data['patient_id'] == pid]
        plot_data = pid_data[['patient_id', 'time_point_days', analyte,
                              '{}_dilution'.format(analyte),
                              '{}_max_dilution'.format(analyte)]]
        # clean strings to remove '>'/'<', convert 'fail' to NaN
        plot_data[analyte] = plot_data[analyte].apply(clean_strings)
        # convert strings to float
        plot_data[analyte] = plot_data[analyte].apply(float)
        # only keep non-null values
        plot_data = plot_data.loc[~plot_data[analyte].isnull()]
        # get all distinct time point values
        first_days = plot_data['time_point_days'].unique().tolist()
        first_days.sort()
        # save the first three time point values
        first_days = first_days[:3]
        # save the initial time point value
        day_zero = first_days[0]
        # save the inital analyte value, at the initial time point value
        initial_pg = plot_data.loc[plot_data['time_point_days'] == day_zero, analyte].item()
        # take log of the floats
        plot_data[analyte] = plot_data[analyte].apply(math.log)
        # plot each patient_id separately
        f = plt.figure()
        f.add_subplot()
        # logic for splitting into groups, default group is blue
        plt_color = 'blue'
        # if initial value too small set group to black
        if initial_pg < 10:
            plt_color = 'black'
        # otherwise...
        else:
            early_values = plot_data.loc[plot_data['time_point_days'].isin(first_days)]
            # save the mean of the first three time point values
            mean_val = early_values[analyte].mean()
            # exclude the first three time point values from testing
            sub_plot = plot_data.loc[~plot_data['time_point_days'].isin(first_days)]
            sub_val = sub_plot[analyte].values
            # For each time point beyond the third in a given sample:
            # 1) If the value at that point is greater than the mean of the values at the first three time points and
            # 2) If
            #   - the difference between the immediately preceding value and the immediately following value is
            #   greater than 1% of the preceding value OR
            #   - the immediately following value is greater than 80% of the mean of the values at the first three
            #   time points
            # and
            # 3) If the difference between the value and the mean of the values at the first three time points is
            # greater than 1% of that mean and
            # 4) If the immediately following value is greater than 80% of the mean of the values at the first three
            # time points, then
            # 5) The patient ID is set to group red
            for i in range(len(sub_val)):
                try:
                    if mean_val < sub_val[i]:
                        cond1 = abs(sub_val[i + 1] - sub_val[i - 1]) > (.01 * sub_val[i - 1])
                        cond2 = sub_val[i + 1] > .8 * mean_val
                        if cond1 or cond2:
                            if (sub_val[i] - mean_val) > (.01 * mean_val):
                                if sub_val[i + 1] > .9 * mean_val:
                                    plt_color = 'red'
                except IndexError:
                    pass
        # plot the data with the selected group expressed as plot color
        plt.plot(plot_data['time_point_days'], plot_data[analyte],
                 color=plt_color)
        # label the plot and the axes
        true_analyte = analyte_name[0]
        units = analyte_name[1]
        title = "Analyte: {}, Patient_id: {}".format(true_analyte, pid)
        plt.xlabel("Timepoint, in days")
        plt.ylabel("Log of analyte value, {}".format(units))
        plt.title(title)
        plt.tight_layout()
        pp.savefig(f)
        plt.close()
    pp.close()


def main(run_shapes, run_points, run_connected, run_hrp2):
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
        if run_hrp2 and (analyte == 'HRP2_pg_ml'):
            hrp2_grouping(main_data, analyte, analyte_name)


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
