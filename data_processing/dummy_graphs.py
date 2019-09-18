import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
pd.set_option('chained_assignment', None)


def main():
    time_series = PdfPages('C:/Users/lzoeckler/Desktop/sample_timeseries.pdf')

    time_point_days = [1, 2, 3, 4, 5, 6, 7, 8]
    line_val = 4
    plt.rc('xtick', labelsize=24)
    plt.rc('ytick', labelsize=24)

    f, ax1 = plt.subplots()
    title = 'Chronic asymptomatic infection'
    hrp2 = [3.3, 2.6, 2.4, 2.3, 2.2, 2.2, 2.2, 2.2]
    pldh = [3.3, 2.2, 2, 1.9, 1.8, 1.8, 1.8, 1.8]
    ax1.plot(time_point_days, hrp2, c='black', alpha=0.6, label='HRP2')
    ax1.plot(time_point_days, pldh, c='green', alpha=0.6, label='pLDH')
    ax1.plot(np.array([line_val, line_val]), np.array([-1, 8]), color='brown',
             linestyle='--', alpha=0.6)
    ax1.set_title(title, fontsize=24)
    ax1.set_xlabel('Time (days)', fontsize=24)
    ax1.set_ylabel('Antigen concentration\n(log10)', fontsize=20)
    ax1.set_ylim(0, 5)
    ax1.set_xlim(0.8, 7)
    f.tight_layout()
    time_series.savefig(f)
    plt.close()

    f, ax1 = plt.subplots()
    title = 'Current symptomatic infection'
    hrp2 = [4.3, 3.8, 4.1, 3.9, 3.6, 4, 4.1, 3.9]
    pldh = [4.2, 3.3, 4, 4.1, 3.8, 3.7, 3.9, 4]
    ax1.plot(time_point_days, hrp2, c='black', alpha=0.6, label='HRP2')
    ax1.plot(time_point_days, pldh, c='green', alpha=0.6, label='pLDH')
    ax1.plot(np.array([line_val, line_val]), np.array([-1, 8]), color='brown',
             linestyle='--', alpha=0.6)
    ax1.set_title(title, fontsize=24)
    ax1.set_xlabel('Time (days)', fontsize=24)
    ax1.set_ylabel('Antigen concentration\n(log10)', fontsize=20)
    ax1.set_ylim(0, 5)
    ax1.set_xlim(0.8, 7)
    f.tight_layout()
    time_series.savefig(f)
    plt.close()

    f, ax1 = plt.subplots()
    title = 'Recently treated infection'
    hrp2 = [4.3, 3, 2.5, 2.3, 2.2, 2.1, 2, 1.9]
    pldh = [4.3, 1.2, 0.9, 0.89, 0.88, 0.87, 0.87, 0.87]
    ax1.plot(time_point_days, hrp2, c='black', alpha=0.6, label='HRP2')
    ax1.plot(time_point_days, pldh, c='green', alpha=0.6, label='pLDH')
    ax1.plot(np.array([line_val, line_val]), np.array([-1, 8]), color='brown',
             linestyle='--', alpha=0.6)
    ax1.set_title(title, fontsize=24)
    ax1.set_xlabel('Time (days)', fontsize=24)
    ax1.set_ylabel('Antigen concentration\n(log10)', fontsize=20)
    ax1.set_ylim(0, 5)
    ax1.set_xlim(0.8, 7)
    f.tight_layout()
    time_series.savefig(f)
    plt.close()

    f, ax1 = plt.subplots()
    hrp2 = [4.3, 3, 2.5, 2.3, 2.2, 2.1, 2, 1.9]
    pldh = [4.3, 1.2, 0.9, 0.89, 0.88, 0.87, 0.87, 0.87]
    ln4 = ax1.plot(time_point_days, hrp2, c='black', alpha=0.6, label='HRP2')
    ln3 = ax1.plot(time_point_days, pldh, c='green', alpha=0.6, label='pLDH')
    ax1.set_ylim(0, 5)
    ax1.set_xlim(0.8, 7)
    lns = ln4 + ln3
    lns = lns + [Line2D([0], [0], color='brown', linestyle='--',
                        label='Time of survey', alpha=0.6)]
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, bbox_to_anchor=(.69, -.2), ncol=2, fontsize=14)
    f.tight_layout()
    time_series.savefig(f)
    plt.close()


if __name__ == '__main__':
    main()
