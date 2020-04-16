import glob
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# import helper functions
from threshold_helpers import clean_strings


def main(input_dir, output_dir, lower_threshold, upper_threshold, hrp2,
         hrp2_lower_threshold, hrp2_upper_threshold, run_single, run_dual, show):
    # Convert pg values to ng for labeling
    ng_lower = lower_threshold / 1000
    ng_upper = upper_threshold / 1000
    h_ng_lower = hrp2_lower_threshold / 1000
    h_ng_upper = hrp2_upper_threshold / 1000

    # Read in all CSVs
    all_files = glob.glob('{}/*.csv'.format(input_dir))
    big_df = []
    for file in all_files:
        df = pd.read_csv(file)
        big_df.append(df)
    big_df = pd.concat(big_df, sort=True)
    # Subset to CSVs where PCR results are non-null
    big_df = big_df.loc[~big_df['PCR_pos'].isnull()]

    # List out columns that need to be cleaned
    val_cols = ['quansys_HRP2_pg_ml', 'quansys_LDH_Pan_pg_ml', 'quansys_LDH_Pv_pg_ml',
                'quansys_LDH_Pf_pg_ml', 'quansys_CRP_ng_ml']

    # Subset dataframe to just positive PCR cases
    pos_df = big_df.loc[big_df['PCR_pos'] == 1]
    # Drop rows with null pLDH or HRP2
    pos_df = pos_df.loc[~pos_df['quansys_LDH_Pv_pg_ml'].isnull()]
    pos_df = pos_df.loc[~pos_df['quansys_HRP2_pg_ml'].isnull()]
    # Clean columns listed above
    pos_df[val_cols] = pos_df[val_cols].applymap(clean_strings)

    # Copy positive dataframe to manipulate
    den_df = pos_df.copy(deep=True)
    # Subset copy to only cleaned columns, PCR status, and sample_id
    den_df = den_df[val_cols + ['PCR_pos', 'sample_id']]

    # Copy positive dataframe to manipulate
    feb_df = pos_df.copy(deep=True)
    # Subset copy to febrile cases
    feb_df = feb_df.loc[feb_df['febrile'] == 1]
    # Create a new dataframe with only Pf+ cases
    feb_pf_df = feb_df.loc[feb_df['pf'] == 1]
    # Subset to only required columns
    feb_pf_df = feb_pf_df[val_cols + ['PCR_pos', 'sample_id']]
    # Create a new dataframe with only Pv+ cases
    feb_pv_df = feb_df.loc[feb_df['pv'] == 1]
    # Subset to only required columns
    feb_pv_df = feb_pv_df[val_cols + ['PCR_pos', 'sample_id']]
    # Subset entire febrile case dataframe to only required columns
    feb_df = feb_df[val_cols + ['PCR_pos', 'sample_id']]

    # Copy positive dataframe to manipulate
    non_feb_df = pos_df.copy(deep=True)
    # Subset copy to non-febrile cases
    non_feb_df = non_feb_df.loc[non_feb_df['febrile'] == 0]
    # Create a new dataframe with only Pf+ cases
    non_feb_pf_df = non_feb_df.loc[non_feb_df['pf'] == 1]
    # Subset to only required columns
    non_feb_pf_df = non_feb_pf_df[val_cols + ['PCR_pos', 'sample_id']]
    # Create a new dataframe with only Pv+ cases
    non_feb_pv_df = non_feb_df.loc[non_feb_df['pv'] == 1]
    # Subset to only required columns
    non_feb_pv_df = non_feb_pv_df[val_cols + ['PCR_pos', 'sample_id']]
    # Subset entire non-febrile case dataframe to only required columns
    non_feb_df = non_feb_df[val_cols + ['PCR_pos', 'sample_id']]

    # Copy positive dataframe to manipulate
    pf_df = pos_df.copy(deep=True)
    # Subset copy to Pf+ cases
    pf_df = pf_df.loc[pf_df['pf'] == 1]
    # Subset to only required columns
    pf_df = pf_df[val_cols + ['PCR_pos', 'sample_id']]

    # Copy positive dataframe to manipulate
    pv_df = pos_df.copy(deep=True)
    # Subset copy to Pv+ cases
    pv_df = pv_df.loc[pv_df['pv'] == 1]
    # Subset to only required columns
    pv_df = pv_df[val_cols + ['PCR_pos', 'sample_id']]

    # Run the following code only if single graphs (with one density line) are desired
    if run_single:
        print('Running single plots')
        # Collect all dataframes generated above and associate them with a little metadata
        value_list = [(den_df, 'PCR+ pLDH', 'density_plot'), (feb_df, 'Febrile PCR+ pLDH', 'febrile_plot'),
                      (feb_pf_df, 'Febrile Pf+ PCR+ pLDH', 'febrile_pf_plot'),
                      (feb_pv_df, 'Febrile Pv+ PCR+ pLDH', 'febrile_pv_plot'),
                      (non_feb_df, 'Non-febrile PCR+ pLDH', 'non_febrile_plot'),
                      (non_feb_pf_df, 'Non-febrile Pf+ PCR+ pLDH', 'non_febrile_pf_plot'),
                      (non_feb_pv_df, 'Non-febrile Pv+ PCR+ pLDH', 'non_febrile_pv_plot'),
                      (pf_df, 'Pf+ PCR+ pLDH', 'pf_plot'), (pv_df, 'Pv+ PCR+ pLDH', 'pv_plot')]

        # Loop over collected dataframes and metadata
        for test_df, pdf_title, plot_title in value_list:
            # Get density distribution of pLDH values
            x = test_df['quansys_LDH_Pan_pg_ml'].values
            # Create a subset where pLDH is inbetween the lower and upper thresholds and find the length of the subset
            sub = test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)]
            sub = sub.loc[sub['quansys_LDH_Pan_pg_ml'] < np.log10(upper_threshold)]
            sub = len(sub)
            # Create a subset where pLDH is below the lower threshold, find its length
            below = len(test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] < np.log10(lower_threshold)])
            # Create a subset where pLDH is above the lower threshold, find its length
            above = len(test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] > np.log10(lower_threshold)])
            # Create a subset where pLDH is above the upper threshold, find its length
            ten_above = len(test_df.loc[test_df['quansys_LDH_Pan_pg_ml'] > np.log10(upper_threshold)])
            # Find the length of all samples in the given dataframe
            total = len(test_df)
            # Create an extremely complicated title using the data obtained above, also calculate percent of each
            # subset to total
            title = '''{p}\n{s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)
            {b}/{t} samples fall below {lt} ng/ml (~{bt}%)\n{a}/{t} samples fall above {lt} ng/ml(~{at}%)
            {ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(p=pdf_title, t=total, s=sub, lt=ng_lower,
                                                                       ut=ng_upper, b=below, a=above, ta=ten_above,
                                                                       st=np.round(100 * (sub / total), 1),
                                                                       bt=np.round(100 * (below / total), 1),
                                                                       at=np.round(100 * (above / total), 1),
                                                                       tat=np.round(100 * (ten_above / total), 1))
            # The rest of this script is all generating graphs for output
            pp = PdfPages('{}/{}_{}ng_{}ng.pdf'.format(output_dir, plot_title, ng_lower, ng_upper))
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
            if show:
                plt.show()
            pp.savefig(f)
            plt.close()
            pp.close()

    # Run the following code only if dual graphs (with two density lines, febrile and non-febrile) are desired
    if run_dual:
        print('Running dual plots')
        for feb_df, non_feb_df, species in [(feb_df, non_feb_df, 'Pf&Pv'), (feb_pf_df, non_feb_pf_df, 'Pf'),
                                            (feb_pv_df, non_feb_pv_df, 'Pv')]:
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
{s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)\n{b}/{t} samples fall below {lt} ng/ml (~{bt}%)
{a}/{t} samples fall above {lt} ng/ml(~{at}%)\n{ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(sp=species,
                                                                        t=total, s=sub, lt=ng_lower, ut=ng_upper,
                                                                        a=above, ta=ten_above, b=below,
                                                                        st=np.round(100 * (sub / total), 1),
                                                                        bt=np.round(100 * (below / total), 1),
                                                                        at=np.round(100 * (above / total), 1),
                                                                        tat=np.round(100 * (ten_above / total), 1))

            pp = PdfPages('{}/combined_{}_{}ng_{}ng.pdf'.format(output_dir, species, ng_lower, ng_upper))
            ax = sns.distplot(x2, hist=False, color='k', label='Non-febrile, {}'.format(species))
            ln = ax.lines[0]
            y = ln.get_ydata()
            x2a = ln.get_xdata()
            ax.fill_between(x2a, 0, y, where=(0.05 > x2a) & (x2a > -10), color='purple',
                            label='Samples below LoD')
            sns.distplot(x1, hist=False, color='b', label='Febrile, {}'.format(species))
            ax.axvline(np.log10(lower_threshold), color='r')
            ax.axvline(np.log10(upper_threshold), color='g')
            ax.set_ylim(0, 0.5)
            ax.legend()
            ax.set_ylabel('Density')
            ax.set_xlabel(title2, fontsize=12)
            plt.xticks([-1, 0, 1, 2, 3, 4, 6, 8], [0.1, 1, 10, 100, 1000, 10000, 1000000, 100000000])
            f = ax.get_figure()
            plt.title(title1)
            plt.tight_layout()
            if show:
                plt.show()
            pp.savefig(f)
            plt.close()
            pp.close()

    if hrp2:
        print('Running HRP2 plots')
        for feb_df, non_feb_df, species in [(feb_pf_df, non_feb_pf_df, 'Pf')]:
            x1 = feb_df['quansys_HRP2_pg_ml'].values
            x2 = non_feb_df['quansys_HRP2_pg_ml'].values

            sub = feb_df.loc[feb_df['quansys_HRP2_pg_ml'] > np.log10(hrp2_lower_threshold)]
            sub = sub.loc[sub['quansys_HRP2_pg_ml'] < np.log10(hrp2_upper_threshold)]
            sub = len(sub)
            below = len(feb_df.loc[feb_df['quansys_HRP2_pg_ml'] < np.log10(hrp2_lower_threshold)])
            above = len(feb_df.loc[feb_df['quansys_HRP2_pg_ml'] > np.log10(hrp2_lower_threshold)])
            ten_above = len(feb_df.loc[feb_df['quansys_HRP2_pg_ml'] > np.log10(hrp2_upper_threshold)])
            total = len(feb_df)

            title1 = '''Febrile {sp}+:\n{s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)
{b}/{t} samples fall below {lt} ng/ml (~{bt}%)\n{a}/{t} samples fall above {lt} ng/ml(~{at}%)
{ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(sp=species, s=sub, t=total, lt=h_ng_lower,
            ut=h_ng_upper, b=below, ta=ten_above, a=above, st=np.round(100 * (sub / total), 1),
            bt=np.round(100 * (below / total), 1), at=np.round(100 * (above / total), 1),
            tat=np.round(100 * (ten_above / total), 1))

            sub = non_feb_df.loc[non_feb_df['quansys_HRP2_pg_ml'] > np.log10(hrp2_lower_threshold)]
            sub = sub.loc[sub['quansys_HRP2_pg_ml'] < np.log10(hrp2_upper_threshold)]
            sub = len(sub)
            below = len(non_feb_df.loc[non_feb_df['quansys_HRP2_pg_ml'] < np.log10(hrp2_lower_threshold)])
            above = len(non_feb_df.loc[non_feb_df['quansys_HRP2_pg_ml'] > np.log10(hrp2_lower_threshold)])
            ten_above = len(non_feb_df.loc[non_feb_df['quansys_HRP2_pg_ml'] > np.log10(hrp2_upper_threshold)])
            total = len(non_feb_df)

            title2 = '''log10 HRP2 pg/ml\n\nNon-febrile {sp}+:
{s}/{t} samples fall between {lt} and {ut} ng/ml (~{st}%)\n{b}/{t} samples fall below {lt} ng/ml (~{bt}%)
{a}/{t} samples fall above {lt} ng/ml(~{at}%)\n{ta}/{t} samples fall above {ut} ng/ml (~{tat}%)'''.format(
            sp=species, t=total, s=sub, lt=h_ng_lower, ut=h_ng_upper, a=above, ta=ten_above, b=below,
            st=np.round(100 * (sub / total), 1), bt=np.round(100 * (below / total), 1),
            at=np.round(100 * (above / total), 1),
            tat=np.round(100 * (ten_above / total), 1))

            pp = PdfPages('{}/HRP2_combined_{}_{}ng_{}ng.pdf'.format(output_dir, species, h_ng_lower, h_ng_upper))
            ax = sns.distplot(x2, hist=False, color='k', label='Non-febrile, {}'.format(species))
            ln = ax.lines[0]
            y = ln.get_ydata()
            x2a = ln.get_xdata()
            ax.fill_between(x2a, 0, y, where=(0.05 > x2a) & (x2a > -10), color='purple', label='Samples below LoD')
            sns.distplot(x1, hist=False, color='b', label='Febrile, {}'.format(species))
            ax.axvline(np.log10(hrp2_lower_threshold), color='r')
            ax.axvline(np.log10(hrp2_upper_threshold), color='g')
            ax.set_ylim(0, 0.5)
            ax.legend()
            ax.set_ylabel('Density')
            ax.set_xlabel(title2, fontsize=12)
            plt.xticks([-1, 0, 1, np.log10(80), np.log10(400), 4, 6, 8],
                       [0.1, 1, 10, 80, 400, 10000, 1000000, 100000000])
            f = ax.get_figure()
            plt.title(title1)
            plt.tight_layout()
            if show:
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
    parser.add_argument('-hr', '--hrp2', action='store_true',
                        help='Produce HRP2 graph')
    parser.add_argument('-hlt', '--hrp2_lower_threshold', type=int,
                        default=80, help='Lower threshold, in pg, for HRP2 graph')
    parser.add_argument('-hut', '--hrp2_upper_threshold', type=int,
                        default=400, help='Lower threshold, in pg, for HRP2 graph')
    parser.add_argument('-rs', '--run_single', action='store_true',
                        help='Produce single density graphs')
    parser.add_argument('-rd', '--run_dual', action='store_true',
                        help='Produce dual density graphs')
    parser.add_argument('-sh', '--show', action='store_true',
                        help='Show graphs as they run')
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir, lower_threshold=args.lower_threshold,
         upper_threshold=args.upper_threshold, hrp2=args.hrp2, hrp2_lower_threshold=args.hrp2_lower_threshold,
         hrp2_upper_threshold=args.hrp2_upper_threshold, run_single=args.run_single, run_dual=args.run_dual,
         show=args.show)
