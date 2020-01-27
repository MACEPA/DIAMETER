import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import helper functions
from threshold_helpers import clean_strings


def main(input_dir, output_dir, lower_threshold, upper_threshold, run_single, run_dual):
    ng_lower = lower_threshold / 100
    ng_upper = upper_threshold / 100

    all_files = glob.glob('{}/*.csv'.format(input_dir))
    big_df = []
    for file in all_files:
        df = pd.read_csv(file)
        big_df.append(df)
    big_df = pd.concat(big_df, sort=True)
    big_df = big_df.loc[~big_df['PCR_pos'].isnull()]

    val_cols = ['quansys_HRP2_pg_ml', 'quansys_LDH_Pan_pg_ml', 'quansys_LDH_Pv_pg_ml',
                'quansys_LDH_Pf_pg_ml', 'quansys_CRP_ng_ml']

    pos_df = big_df.loc[big_df['PCR_pos'] == 1]
    pos_df = pos_df.loc[~pos_df['quansys_LDH_Pv_pg_ml'].isnull()]
    pos_df = pos_df.loc[~pos_df['quansys_HRP2_pg_ml'].isnull()]
    pos_df[val_cols] = pos_df[val_cols].applymap(clean_strings)

    den_df = pos_df.copy(deep=True)
    den_df = den_df[val_cols + ['PCR_pos', 'sample_id']]

    feb_df = pos_df.copy(deep=True)
    feb_df = feb_df.loc[feb_df['febrile'] == 1]
    feb_pf_df = feb_df.loc[feb_df['pf'] == 1]
    feb_pv_df = feb_df.loc[feb_df['pv'] == 1]
    feb_pf_df = feb_pf_df[val_cols + ['PCR_pos', 'sample_id']]
    feb_pv_df = feb_pv_df[val_cols + ['PCR_pos', 'sample_id']]
    feb_df = feb_df[val_cols + ['PCR_pos', 'sample_id']]

    non_feb_df = pos_df.copy(deep=True)
    non_feb_df = non_feb_df.loc[non_feb_df['febrile'] == 0]
    non_feb_pf_df = non_feb_df.loc[non_feb_df['pf'] == 1]
    non_feb_pv_df = non_feb_df.loc[non_feb_df['pv'] == 1]
    non_feb_pf_df = non_feb_pf_df[val_cols + ['PCR_pos', 'sample_id']]
    non_feb_pv_df = non_feb_pv_df[val_cols + ['PCR_pos', 'sample_id']]
    non_feb_df = non_feb_df[val_cols + ['PCR_pos', 'sample_id']]

    pf_df = pos_df.copy(deep=True)
    pf_df = pf_df.loc[pf_df['pf'] == 1]
    pf_df = pf_df[val_cols + ['PCR_pos', 'sample_id']]

    pv_df = pos_df.copy(deep=True)
    pv_df = pv_df.loc[pv_df['pv'] == 1]
    pv_df = pv_df[val_cols + ['PCR_pos', 'sample_id']]

    if run_single:
        value_list = [(den_df, 'PCR+ pLDH', 'density_plot'), (feb_df, 'Febrile PCR+ pLDH', 'febrile_plot'),
                      (feb_pf_df, 'Febrile Pf+ PCR+ pLDH', 'febrile_pf_plot'),
                      (feb_pv_df, 'Febrile Pv+ PCR+ pLDH', 'febrile_pv_plot'),
                      (non_feb_df, 'Non-febrile PCR+ pLDH', 'non_febrile_plot'),
                      (non_feb_pf_df, 'Non-febrile Pf+ PCR+ pLDH', 'non_febrile_pf_plot'),
                      (non_feb_pv_df, 'Non-febrile Pv+ PCR+ pLDH', 'non_febrile_pv_plot'),
                      (pf_df, 'Pf+ PCR+ pLDH', 'pf_plot'), (pv_df, 'Pv+ PCR+ pLDH', 'pv_plot')]

        for test_df, pdf_title, plot_title in value_list:
            x = test_df['quansys_LDH_Pan_pg_ml'].values

            sub = test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)]
            sub = sub.loc[sub['quansys_LDH_Pan_pg_ml'] < np.log10(upper_threshold)]
            sub = len(sub)
            below = len(test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] < np.log10(lower_threshold)])
            above = len(test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)])
            ten_above = len(test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] > np.log10(upper_threshold)])
            total = len(test_df)

            title = '''{p}\n{s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)
            {b}/{t} samples fall below {lt} ng/ml (~{bt}%)\n{a}/{t} samples fall above {lt} ng/ml(~{at}%)
            {ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(p=pdf_title, t=total, s=sub, lt=ng_lower,
                                                                       ut=ng_upper, b=below, a=above, ta=ten_above,
                                                                       st=np.round(100 * (sub / total), 1),
                                                                       bt=np.round(100 * (below / total), 1),
                                                                       at=np.round(100 * (above / total), 1),
                                                                       tat=np.round(100 * (ten_above / total), 1))

            pp = PdfPages('{}/{}.pdf'.format(output_dir, plot_title))
            ax = sns.distplot(x, hist=False, color='k')
            ln = ax.lines[0]
            y = ln.get_ydata()
            x1 = ln.get_xdata()
            ax.fill_between(x1, 0, y, where=(np.log10(upper_threshold) > x1) & (x1 > np.log10(lower_threshold)),
                            color='k', label='Samples between\n{} and {} ng/ml'.format(ng_lower, ng_upper))
            ax.fill_between(x1, 0, y, where=(np.log10(upper_threshold) < x1), color='g',
                            label='Samples above\n{} ng/ml'.format(ng_upper))
            ax.fill_between(x1, 0, y, where=(x1 < np.log10(lower_threshold,)), color='r',
                            label='Samples below\n{} ng/ml'.format(ng_lower))
            ax.set_ylim(0, 0.6)
            ax.legend()
            ax.set_ylabel('Density')
            ax.set_xlabel('log10 pLDH pg/ml')
            plt.xticks([-1, 0, 1, 2, 3, 4, 6, 8], [0.1, 1, 10, 100, 1000, 10000, 1000000, 100000000])
            f = ax.get_figure()
            plt.title(title)
            plt.tight_layout()
            plt.show()
            pp.savefig(f)
            plt.close()
            pp.close()

        if run_dual:
            for feb_df, non_feb_df, species in [(feb_pf_df, non_feb_pf_df, 'Pf'), (feb_pv_df, non_feb_pv_df, 'Pv')]:
                x1 = feb_df['quansys_LDH_Pan_pg_ml'].values
                x2 = non_feb_df['quansys_LDH_Pan_pg_ml'].values

                sub = feb_df.loc[feb_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)]
                sub = sub.loc[sub['quansys_LDH_Pan_pg_ml'] < np.log10(upper_threshold)]
                sub = len(sub)
                below = len(feb_df.loc[feb_df['quansys_LDH_Pan_pg_ml'] < np.log10(lower_threshold)])
                above = len(feb_df.loc[feb_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)])
                ten_above = len(feb_df.loc[feb_df['quansys_LDH_Pan_pg_ml'] > np.log10(upper_threshold)])
                total = len(feb_df)

                title1 = '''Febrile {sp}+:\n{s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)
                {b}/{t} samples fall below {lt} ng/ml (~{bt}%)\n{a}/{t} samples fall above {lt} ng/ml(~{at}%)
                {ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(sp=species, s=sub, t=total, lt=ng_lower,
                                                                           ut=ng_upper, b=below, ta=ten_above, a=above,
                                                                           st=np.round(100 * (sub / total), 1),
                                                                           bt=np.round(100 * (below / total), 1),
                                                                           at=np.round(100 * (above / total), 1),
                                                                           tat=np.round(100 * (ten_above / total), 1))

                sub = non_feb_df.loc[non_feb_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)]
                sub = sub.loc[sub['quansys_LDH_Pan_pg_ml'] < np.log10(upper_threshold)]
                sub = len(sub)
                below = len(non_feb_df.loc[non_feb_df['quansys_LDH_Pan_pg_ml'] < np.log10(lower_threshold)])
                above = len(non_feb_df.loc[non_feb_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)])
                ten_above = len(non_feb_df.loc[non_feb_df['quansys_LDH_Pan_pg_ml'] > np.log10(upper_threshold)])
                total = len(non_feb_df)

                title2 = '''log10 pLDH pg/ml\n\nNon-febrile {sp}+:
                {s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)
                {b}/{t} samples fall below {lt} ng/ml (~{bt}%)\n{a}/{t} samples fall above {lt} ng/ml(~{at}%)
                {ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(sp=species, t=total, s=sub, lt=ng_lower,
                                                                           ut=ng_upper, a=above, ta=ten_above, b=below,
                                                                           st=np.round(100 * (sub / total), 1),
                                                                           bt=np.round(100 * (below / total), 1),
                                                                           at=np.round(100 * (above / total), 1),
                                                                           tat=np.round(100 * (ten_above / total), 1))

                pp = PdfPages('{}/combined_{}.pdf'.format(output_dir, species))
                ax = sns.distplot(x1, hist=False, color='b', label='Febrile')
                sns.distplot(x2, hist=False, color='k', label='Non-febrile')
                ax.axvline(np.log10(lower_threshold), color='r')
                ax.axvline(np.log10(upper_threshold), color='g')
                ax.set_ylim(0, 0.6)
                ax.legend()
                ax.set_ylabel('Density')
                ax.set_xlabel(title2, fontsize=12)
                plt.xticks([-1, 0, 1, 2, 3, 4, 6, 8], [0.1, 1, 10, 100, 1000, 10000, 1000000, 100000000])
                f = ax.get_figure()
                plt.title(title1)
                plt.tight_layout()
                plt.show()
                pp.savefig(f)
                plt.close()
                pp.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop/all_data/separated/binned',
                        help='Input directory for all PCR data')
    parser.add_argument('-od', '--output_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop',
                        help='Output directory for PDFs')
    parser.add_argument('-lt', '--lower_threshold', type=int,
                        default=1000, help='Lower threshold, in pg')
    parser.add_argument('-ut', '--upper_threshold', type=int,
                        default=10000, help='Upper threshold, in pg')
    parser.add_argument('-rs', '--run_single', action='store_true',
                        help='Whether or not to produce single density graphs')
    parser.add_argument('-rd', '--run_dual', action='store_true',
                        help='Whether or not to produce dual density graphs')
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, lower_threshold=args.lower_threshold,
         upper_threshold=args.upper_threshold, run_single=args.run_single, run_dual=args.run_dual)
