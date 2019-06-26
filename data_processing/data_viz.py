import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from data_processing.data_viz_helpers import clean_strings


def main():
    # read in formatted dilution CSV
    main_data = pd.read_csv('C:/Users/lzoeckler/Desktop/4plex/output_data/final_dilutions.csv')
    # set list of analytes for 4plex
    analytes = ['HRP2_pg_ml', 'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml']
    # associate colors and shapes to different dilution values
    all_colors = cm.rainbow(np.linspace(0, 1, 8))
    all_dilutions = ['1', '50', '2500', '125000', '6250000', '312500000', '15625000000', '781250000000']
    all_shapes = ['+', 'v', 's', 'p', 'd', '^', '.', '*']
    combo = zip(all_dilutions, all_colors)
    color_dict = {dil: val for dil, val in combo}
    color_dict['fail'] = np.array([0.0, 0.0, 0.0, 0.0])
    shape_combo = zip(all_dilutions, all_shapes)
    shape_dict = {dil: val for dil, val in shape_combo}
    # loop through analytes to create different PDFs for each
    for analyte in analytes:
        pp = PdfPages('C:/Users/lzoeckler/Desktop/4plex/output_data/{}_graphs.pdf'.format(analyte))
        # create individual graphs for each patient_id
        for pid in main_data['patient_id'].unique():
            # subset data
            pid_data = main_data.loc[main_data['patient_id'] == pid]
            plot_data = pid_data[['patient_id', 'time_point_days', analyte, '{}_dilution'.format(analyte),
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
            ax = f.add_subplot()
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
                color = color_dict[group]
                max_dil = str(int(max_dil))
                shape = shape_dict[max_dil]
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
            title = "analyte: {}, patient_id: {}".format(analyte, pid)
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
        pp.close()


if __name__ == '__main__':
    main()
