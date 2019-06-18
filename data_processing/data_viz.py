import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from data_processing.data_viz_helpers import clean_strings


def main():
    # read in formatted dilution CSV
    main_data = pd.read_csv('C:/Users/lzoeckler/Desktop/4plex/output_data/final_dilutions_time.csv')
    # set list of analytes for 4plex
    analytes = ['HRP2_pg_ml', 'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml']
    # associate colors to different dilution values
    all_colors = cm.rainbow(np.linspace(0, 1, 6))
    all_dilutions = ['1', '50', '2500', '125000', '6250000', '312500000']
    combo = zip(all_dilutions, all_colors)
    color_dict = {dil: val for dil, val in combo}
    color_dict['fail'] = np.array([0.0, 0.0, 0.0, 0.0])
    # loop through analytes to create different PDFs for each
    for analyte in analytes:
        pp = PdfPages('C:/Users/lzoeckler/Desktop/4plex/output_data/{}_color_graphs_ugly.pdf'.format(analyte))
        # create individual graphs for each patient_id
        for pid in main_data['patient_id'].unique():
            # subset data
            pid_data = main_data.loc[main_data['patient_id'] == pid]
            plot_data = pid_data[['patient_id', 'time', analyte, '{}_dilution'.format(analyte),
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
            plt.plot(plot_data['time'], plot_data[analyte], color=plt_color, alpha=0.3)
            dil_vals = plot_data['{}_dilution'.format(analyte)].tolist()
            dil_vals = [str(val) if val != 'fail' else val for val in dil_vals]
            vals = plot_data[analyte].tolist()
            time = plot_data['time'].tolist()
            # return the maximum dilution available for each data point
            maximum = plot_data['{}_max_dilution'.format(analyte)]
            data = zip(time, vals)
            # plot the data points in a scatter with color indicating dilution
            # also label the points with the max dilution available
            for data, group, text in zip(data, dil_vals, maximum):
                x, y = data
                color = color_dict[group]
                plt.scatter(x, y, c=[color], label=group, alpha=1.0)
                ax.annotate(text, (x, y))
            ax.legend()
            hand, labl = ax.get_legend_handles_labels()
            handout = []
            lablout = []
            # fix legend to not duplicate entries
            for h, l in zip(hand, labl):
                if l not in lablout:
                    lablout.append(l)
                    handout.append(h)
            ax.legend(handout, lablout, title='Dilution')
            # plot in log scale
            plt.yscale('log')
            title = "analyte: {}, patient_id: {}".format(analyte, pid)
            plt.title(title)
            plt.tight_layout()
            pp.savefig(f)
        pp.close()


if __name__ == '__main__':
    main()
