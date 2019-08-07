import math
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from matplotlib.lines import Line2D
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
from data_processing.data_viz_helpers import (clean_strings, run_model,
                                              hrp2_complex_grouping,
                                              hrp2_ratio_grouping)
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


def plot_zero_density(main_data):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    density = PdfPages('{}/time_point_density_with_zero.pdf'.format(output_fp))
    good_zero = main_data.loc[main_data['group'] == 'blue']
    good_zero.rename({'time_point_days': 'Group 1 days'}, axis=1, inplace=True)
    good_zero['Group 1 days'].plot.kde()
    bad_zero = main_data.loc[main_data['group'] == 'red']
    bad_zero.rename({'time_point_days': 'Group 2 days'}, axis=1, inplace=True)
    bad_zero['Group 2 days'].plot.kde()
    plt.legend()
    plt.title('Density plot for group 1 vs group 2 days')
    plt.tight_layout()
    density.savefig()
    plt.close()
    density.close()


def plot_nonzero_density(main_data):
    output_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    density = PdfPages('{}/time_point_density_without_zero.pdf'.format(output_fp))
    good_no_zero = main_data.loc[main_data['group'] == 'blue']
    good_no_zero = good_no_zero.loc[good_no_zero['time_point_days'] != 0]
    good_no_zero.rename({'time_point_days': 'Group 1 days'}, axis=1, inplace=True)
    good_no_zero['Group 1 days'].plot.kde()
    bad_no_zero = main_data.loc[main_data['group'] == 'red']
    bad_no_zero = bad_no_zero.loc[bad_no_zero['time_point_days'] != 0]
    bad_no_zero.rename({'time_point_days': 'Group 2 days'}, axis=1, inplace=True)
    bad_no_zero['Group 2 days'].plot.kde()
    plt.legend()
    plt.title('Density plot for group 1 vs group 2 days\n(does not include day 0)')
    plt.tight_layout()
    density.savefig()
    plt.close()
    density.close()


def plot_all_points_analytes(main_data, version):
    ratios = PdfPages('C:/Users/lzoeckler/Desktop/4plex/output_data/{}_good_vs_bad_points.pdf'.format(version))
    pairs = [('HRP2_pg_ml', 'LDH_Pan_pg_ml'), ('HRP2_pg_ml', 'CRP_ng_ml'), ('LDH_Pan_pg_ml', 'CRP_ng_ml')]
    for pair in pairs:
        name1 = ANALYTE_INFO[pair[0]][0]
        name2 = ANALYTE_INFO[pair[1]][0]
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
        # good info
        good_df = main_data.loc[main_data['group'] == 'blue']
        time, pred, coef = run_model(good_df, pair)
        title = '"Good" points\nk = {}'.format(round(coef, 4))
        plt.subplot(1, 2, 1)
        plt.scatter(good_df[pair[0]], good_df[pair[1]], color=good_df['group'], alpha=0.6)
        plt.plot(time, pred, color='black')
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(name1)
        plt.ylabel(name2)
        plt.tight_layout()
        # bad info
        bad_df = main_data.loc[main_data['group'] == 'red']
        time, pred, coef = run_model(bad_df, pair)
        title = '"Bad" points\nk = {}'.format(round(coef, 4))
        plt.subplot(1, 2, 2)
        plt.scatter(bad_df[pair[0]], bad_df[pair[1]], color=bad_df['group'], alpha=0.6)
        plt.plot(time, pred, color='black')
        plt.title(title)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(name1)
        plt.tight_layout()
        ratios.savefig(f)
        plt.show()
        plt.close()
    ratios.close()


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
            plot_hrp2_groups(grouped_data, 'complex')
            plot_all_points_analytes(grouped_data, 'complex')
        if run_ratio_hrp2:
            grouped_data = hrp2_ratio_grouping(rebuilt_data)
            plot_hrp2_groups(grouped_data, 'ratio_based')
            plot_zero_density(grouped_data)
            plot_nonzero_density(grouped_data)
            plot_all_points_analytes(grouped_data, 'ratio_based')


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
