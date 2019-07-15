import math
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
    plot_data = main_data[['patient_id', 'time_point_days', analyte,
                           '{}_dilution'.format(analyte),
                           '{}_max_dilution'.format(analyte)]]
    # clean strings to remove '>'/'<', convert 'fail' to NaN
    plot_data[analyte] = plot_data[analyte].apply(clean_strings)
    # convert strings to float
    plot_data[analyte] = plot_data[analyte].apply(float)
    # take log of the floats
    plot_data[analyte] = plot_data[analyte].apply(math.log)
    # run a simple linear regression on the logged data
    regr = linear_model.LinearRegression()
    time = plot_data['time_point_days'].values.reshape(-1, 1)
    val = plot_data[analyte].values.reshape(-1, 1)
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


def main():
    # read in formatted dilution CSV
    input_fp = 'C:/Users/lzoeckler/Desktop/4plex/output_data'
    main_data = pd.read_csv('{}/final_dilutions.csv'.format(input_fp))
    # loop through analytes to create different PDFs for each
    for analyte in ANALYTE_INFO.keys():
        analyte_name = ANALYTE_INFO[analyte]
        # produce analyte shape graphs
        analyte_shapes(main_data, analyte, analyte_name)
        # produce analyte individual graphs with trend lines, unconnected
        analyte_point_individuals(main_data, analyte, analyte_name)
        # produce analyte individual graphs without trend lines, connected
        analyte_connected_individuals(main_data, analyte, analyte_name)


if __name__ == '__main__':
    main()
