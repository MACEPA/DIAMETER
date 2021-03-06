import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from more_itertools import unique_everseen
from matplotlib.backends.backend_pdf import PdfPages
# import helper functions
from data_viz_helpers import (clean_strings, rebuild_data,
                              hrp2_outliering, hrp2_grouping)
# import constants
from data_viz_helpers import (COLOR_DICT, SHAPE_DICT, ANALYTE_INFO)


def analyte_shapes(main_data, analyte, analyte_name, output_dir):
    pp = PdfPages('{}/{}_graphs.pdf'.format(output_dir, analyte))
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
        f = plt.figure()
        f.add_subplot()
        # line plot with "interest" color
        plt.plot(plot_data['time_point_days'], plot_data[analyte], color='blue', alpha=0.3)
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


def analyte_point_individuals(main_data, analyte, analyte_name, output_dir):
    pp = PdfPages('{}/{}_all_individuals.pdf'.format(output_dir, analyte))
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


def analyte_connected_individuals(main_data, analyte, analyte_name, output_dir):
    pp = PdfPages('{}/{}_all_connected.pdf'.format(output_dir, analyte))
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


def plot_hrp2_groups(main_data, output_dir):
    pp = PdfPages('{}/HRP2_groups.pdf'.format(output_dir))
    for pid in main_data['patient_id'].unique():
        # subset data to individual patient_id
        combo = main_data.loc[main_data['patient_id'] == pid]
        # sort the days to get rid of weird graphing artifacts
        combo = combo.sort_values('time_point_days')
        # fetch the maximum day for help creating threshold lines later
        max_day = max(combo['time_point_days'])
        f, ax1 = plt.subplots()
        title = "patient_id: {}".format(pid)
        ax1.set_title(title)
        # plot the HRP2 against days, the LDH against days, and the different
        # HRP2 points with colors according to group
        ln1 = ax1.plot(combo['time_point_days'], combo['HRP2_pg_ml'],
                       c='black', alpha=0.6, label='HRP2')
        ln2 = ax1.plot(combo['time_point_days'], combo['LDH_Pan_pg_ml'], c='green', alpha=0.6,
                       label='pLDH')
        ax1.scatter(combo['time_point_days'], combo['HRP2_pg_ml'],
                    c=combo['group'])
        # plot the HRP2 threshold for flagging
        ln3 = ax1.plot(np.array([0, max_day]), np.array([4, 4]), linestyle='--',
                       label='HRP2 threshold', alpha=0.6, color='k')
        ax1.set_xlabel('Time point, in days')
        ax1.set_ylabel('Log of pg/ml')
        # clone the y axis for plotting the ratio on a different scale
        ax2 = ax1.twinx()
        # set the new y axis label and color
        ax2.set_ylabel('Ratio of pLDH/HRP2', c='brown')
        # plot the ratio on the new axis
        ax2.plot(combo['time_point_days'], combo['ratio'], c='brown', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='brown')
        # plot the ratio threshold for flagging
        ln4 = ax2.plot(np.array([0, max_day]), np.array([0.8, 0.8]), color='brown',
                       linestyle='--', label='Ratio threshold', alpha=0.6)
        # combine all the lines to create a composite legend
        lns = ln1 + ln2 + ln3 + ln4
        labs = [l.get_label() for l in lns]
        # place the legend below the plot
        ax2.legend(lns, labs, bbox_to_anchor=(.75, -.2), ncol=2)
        f.tight_layout()
        # save the plot
        pp.savefig(f)
        plt.close()
    pp.close()


# CURRENTLY UNUSED
def plot_meta(main_data, output_dir):
    pp = PdfPages('{}/meta_namibia.pdf'.format(output_dir))
    for pid in main_data['patient_id'].unique():
        # subset data to individual patient_id
        combo = main_data.loc[main_data['patient_id'] == pid]
        # sort the days to get rid of weird graphing artifacts
        combo = combo.sort_values('time_point_days')
        # fetch max and min y values for creating returned lines
        max_y = max([max(combo['HRP2_pg_ml']), max(combo['LDH_Pan_pg_ml'])])
        min_y = min([min(combo['HRP2_pg_ml']), min(combo['LDH_Pan_pg_ml'])])
        # fetch different metadata values
        sex = combo['sex'].unique()[0]
        age = combo['age'].unique()[0]
        when_returned = combo['when_returned_with_fever'].unique()[0]
        # clean the date returned value
        if isinstance(when_returned, str):
            try:
                cleaned = [int(when_returned)]
            except ValueError:
                cleaned = when_returned.replace('and ', '')
                cleaned = cleaned.replace(' ', '')
                cleaned = [int(x) for x in cleaned.split(',')]
        else:
            cleaned = []
        # set the date retreated value to int if possible
        when_retreated = combo['when_retreated'].unique()[0]
        if ~np.isnan(when_retreated):
            when_retreated = int(when_retreated)
            # set color of retreated time point to yellow
            combo.loc[combo['time_point_days'] == when_retreated, 'group'] = 'yellow'
        # create the first plot axis
        f, ax1 = plt.subplots()
        title = "pid: {}, sex: {}, age: {}".format(pid, sex, age)
        ax1.set_title(title)
        # plot the HRP2 against days, the LDH against days
        ln1 = ax1.plot(combo['time_point_days'], combo['HRP2_pg_ml'], c='black', alpha=0.6, label='HRP2')
        ln2 = ax1.plot(combo['time_point_days'], combo['LDH_Pan_pg_ml'], c='green', alpha=0.6, label='pLDH')
        ax1.scatter(combo['time_point_days'], combo['HRP2_pg_ml'], c='blue')
        # create lines for returned dates if necessary
        for when_returned in cleaned:
            ax1.plot(np.array([when_returned, when_returned]), np.array([min_y, max_y]),
                     color='purple', linestyle='--', alpha=0.6)
        ax1.set_xlabel('Time point, in days')
        ax1.set_ylabel('Log of pg/ml')
        # clone the y axis for plotting the ratio on a different scale
        ax2 = ax1.twinx()
        # set the new y axis label and color
        ax2.set_ylabel('Ratio of pLDH/HRP2', c='brown')
        # plot the ratio on the new axis
        ln3 = ax2.plot(combo['time_point_days'], combo['ratio'], c='brown', alpha=0.6)
        ax2.tick_params(axis='y', labelcolor='brown')
        # combine first three lines to create a composite legend
        lns = ln1 + ln2 + ln3
        # also add in the flagged points, returned line, and retreated point
        lns = lns + [Line2D([0], [0], marker='o', color='k', label='Flagged', markerfacecolor='r',
                            markersize=10, alpha=0.6)]
        lns = lns + [Line2D([0], [0], color='purple', linestyle='--', label='Returned', alpha=0.6)]
        lns = lns + [Line2D([0], [0], marker='o', color='k', label='Retreated', markerfacecolor='y',
                            markersize=10, alpha=0.6)]
        labs = [l.get_label() for l in lns]
        # function for getting unique list values while preserving order
        labs = list(unique_everseen(labs))
        # place the legend below the plot
        ax2.legend(lns, labs, bbox_to_anchor=(.75, -.2), ncol=2)
        f.tight_layout()
        # save the plot
        pp.savefig(f)
        plt.close()
    pp.close()


# CURRENTLY UNUSED
def plot_zero_density(main_data, output_dir):
    # generate a density plot of time points with both group 1 and group 2
    # this plot includes day 0. the other does not
    # note that these could be bar graphs, but density plots are super easy
    density = PdfPages('{}/time_point_density_with_zero.pdf'.format(output_dir))
    good_zero = main_data.loc[main_data['group'] == 'blue']
    good_zero.rename({'time_point_days': 'Group 1 days'}, axis=1, inplace=True)
    # actual density plot is a one liner
    good_zero['Group 1 days'].plot.kde()
    bad_zero = main_data.loc[main_data['group'] == 'red']
    bad_zero.rename({'time_point_days': 'Group 2 days'}, axis=1, inplace=True)
    # actual density plot is a one liner
    bad_zero['Group 2 days'].plot.kde()
    plt.legend()
    plt.title('Density plot for group 1 vs group 2 days')
    plt.tight_layout()
    density.savefig()
    plt.close()
    density.close()


# CURRENTLY UNUSED
def plot_nonzero_density(main_data, output_dir):
    # generate a density plot of time points with both group 1 and group 2
    # this plot does not include day 0. the other does
    # note that these could be bar graphs, but density plots are super easy
    density = PdfPages('{}/time_point_density_without_zero.pdf'.format(output_dir))
    good_no_zero = main_data.loc[main_data['group'] == 'blue']
    good_no_zero = good_no_zero.loc[good_no_zero['time_point_days'] != 0]
    good_no_zero.rename({'time_point_days': 'Group 1 days'}, axis=1, inplace=True)
    # actual density plot is a one liner
    good_no_zero['Group 1 days'].plot.kde()
    bad_no_zero = main_data.loc[main_data['group'] == 'red']
    bad_no_zero = bad_no_zero.loc[bad_no_zero['time_point_days'] != 0]
    bad_no_zero.rename({'time_point_days': 'Group 2 days'}, axis=1, inplace=True)
    # actual density plot is a one liner
    bad_no_zero['Group 2 days'].plot.kde()
    plt.legend()
    plt.title('Density plot for group 1 vs group 2 days\n(does not include day 0)')
    plt.tight_layout()
    density.savefig()
    plt.close()
    density.close()


def plot_all_points_analytes(main_data, output_dir):
    ratios = PdfPages('{}/good_vs_bad_points.pdf'.format(output_dir, version))
    # create pairs of analytes to iterate over
    pairs = [('HRP2_pg_ml', 'LDH_Pan_pg_ml'), ('HRP2_pg_ml', 'CRP_ng_ml'), ('LDH_Pan_pg_ml', 'CRP_ng_ml')]
    # create a line to use in graphs, for y = x function
    x = np.linspace(0, 10, 1000)
    # iterate over pairs above
    for pair in pairs:
        # fetch the names of each analyte
        name1 = ANALYTE_INFO[pair[0]][0]
        name2 = ANALYTE_INFO[pair[1]][0]
        # set x and y limits for each graph
        if pair == pairs[0]:
            ylim = (1, 9)
        else:
            ylim = (1.4, 6)
        if pair == pairs[2]:
            xlim = (1, 9)
        else:
            xlim = (0, 11)
        f = plt.figure()
        f.add_subplot()
        # subset to the group 1, or "good", points
        good_df = main_data.loc[main_data['group'] == 'blue']
        title = '"Good" points'
        # subset to the first of two plots split horizontally
        plt.subplot(1, 2, 1)
        # scatter all of the group 1 points, with the first analyte on the x axis
        # and the second analyte on the y axis
        plt.scatter(good_df[pair[0]], good_df[pair[1]], color=good_df['group'], alpha=0.6)
        # also plot a y = x line
        plt.plot(x, x, color='black')
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.tight_layout()
        # subset to the group 2, or "bad", points
        bad_df = main_data.loc[main_data['group'] == 'red']
        title = '"Bad" points'
        # subset to the second of two plots split horizontally
        plt.subplot(1, 2, 2)
        # scatter all of the group 2 points, with the first analyte on the x axis
        # and the second analyte on the y axis
        plt.scatter(bad_df[pair[0]], bad_df[pair[1]], color=bad_df['group'], alpha=0.6)
        # also plot a y = x line
        plt.plot(x, x, color='black')
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(name1)
        plt.tight_layout()
        ratios.savefig(f)
        plt.show()
        plt.close()
    ratios.close()


def main(input_dir, output_dir, run_shapes, run_points, run_connected, run_group):
    # read in formatted dilution CSV
    main_data = pd.read_csv('{}/final_dilutions.csv'.format(input_dir))
    # loop through analytes to create different PDFs for each
    for analyte in ANALYTE_INFO.keys():
        analyte_name = ANALYTE_INFO[analyte]
        # produce analyte shape graphs
        if run_shapes:
            analyte_shapes(main_data, analyte, analyte_name, output_dir)
        # produce analyte individual graphs with trend lines, unconnected
        if run_points:
            analyte_point_individuals(main_data, analyte, analyte_name, output_dir)
        # produce analyte individual graphs without trend lines, connected
        if run_connected:
            analyte_connected_individuals(main_data, analyte, analyte_name, output_dir)
        # produce HRP2 groups
    if run_group:
        val_cols = ['HRP2_pg_ml', 'LDH_Pan_pg_ml', 'CRP_ng_ml']
        # generate a dataframe that's easier to work with for HRP2 grouping
        rebuilt_data = rebuild_data(main_data, val_cols)
        # run a grouping based on iterative linear regressions in log space
        # get outliered data
        outliered_data = hrp2_outliering(rebuilt_data)
        # get grouped data
        grouped_data = hrp2_grouping(outliered_data)
        # generate all the different plots
        plot_hrp2_groups(grouped_data, ouput_dir)
        plot_all_points_analytes(grouped_data, output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop/4plex/output_data',
                        help='Input directory for all PCR data')
    parser.add_argument('-od', '--output_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop/4plex/output_data',
                        help='Input directory for all PCR data')
    parser.add_argument('-rs', '--run_shapes', action='store_true',
                        help='Whether or not to produce shape graphs')
    parser.add_argument('-rp', '--run_points', action='store_true',
                        help='Whether or not to produce point graphs')
    parser.add_argument('-rc', '--run_connected', action='store_true',
                        help='Whether or not to produce connected graphs')
    parser.add_argument('-rg', '--run_group', action='store_true',
                        help='Whether or not to produce HRP2 grouping graphs')
    args = parser.parse_args()
    main(input_dir=args.input_dir, run_shapes=args.run_shapes, run_points=args.run_points,
         run_connected=args.run_connected, run_group=args.run_group)
