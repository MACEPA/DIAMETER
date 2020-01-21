import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
# import helper functions
from data_viz_helpers import clean_strings
# set pandas options
pd.set_option('chained_assignment', None)


def main(input_dir):
    # Read in slightly formatted Mali data
    main_data = pd.read_csv('{}/for_viz.csv'.format(input_dir))
    # Set and clean value columns, then put them in log10 space
    val_cols = ['HRP2_pg_ml', 'LDH_Pan_pg_ml', 'LDH_Pv_pg_ml', 'CRP_ng_ml']
    main_data[val_cols] = main_data[val_cols].applymap(clean_strings)
    main_data[val_cols] = main_data[val_cols].applymap(np.log10)
    # Keep only columns of interest
    main_data = main_data[['sample_id', 'id_number', 'HRP2_pg_ml', 'LDH_Pan_pg_ml',
                           'LDH_Pv_pg_ml', 'CRP_ng_ml', 'timepoint_days', 'date_dif',
                           'drug', 'age_yrs', 'RDT_pos', 'HRP2_result',
                           'LDH_Pan_result', 'LDH_Pv_result']]
    main_data.rename(columns={'id_number': 'participant_id'}, inplace=True)
    # Make a subset of only RDT positive results
    pos_vals = main_data.loc[main_data['RDT_pos'] == 1]
    # Set a little red flag for weeeird RDT positives
    pos_vals.loc[pos_vals['HRP2_pg_ml'] < 1.4, 'RDT_pos'] = -999
    # Make a subset of only RDT negative results
    neg_vals = main_data.loc[main_data['RDT_pos'] != 1]
    # Recombine the two subsets, overwriting main_data
    main_data = pd.concat([pos_vals, neg_vals])
    # Color code the different RDT results
    rdt_vars = {'RDT_pos': {0: 'green', 1: 'red', 2: 'yellow', np.nan: 'black'}}
    main_data.replace(rdt_vars, inplace=True)
    # Set timepoint_days to be date_df where it's null
    null_days = main_data.loc[main_data['timepoint_days'].isnull()]
    null_days = null_days.loc[~null_days['drug'].isnull()]
    null_days['timepoint_days'] = null_days['date_dif']
    main_data = main_data.loc[~main_data['timepoint_days'].isnull()]
    main_data = pd.concat([null_days, main_data])
    # Toss out unusable patient ids (also keep track of some weird ones...)
    unusable_pids = [316, 317, 318, 329, 338, 371, 396, 416, 425, 441, 461, 472, 485, 500]
    otherwise_bad_pids = [330]
    questionable_pids = [311, 335, 352, 398, 473, 491, 496, 497]
    usable_data = main_data.loc[~main_data['participant_id'].isin(unusable_pids)]
    usable_data = usable_data.loc[~usable_data['participant_id'].isin(otherwise_bad_pids)]
    # Set the default initial class color to purple, will be overwritten later
    usable_data['class'] = 'purple'
    # Split into 0, 1, and 2 treatment day pids (all 3 treatment day pids are bad, so no need to split them)
    no_tday = []
    one_tday = []
    two_tday = []
    for pid in usable_data['participant_id'].unique():
        pid_data = usable_data.loc[usable_data['participant_id'] == pid]
        null_dates = pid_data.loc[pid_data['date_dif'].isnull()]
        if len(null_dates) == len(pid_data):
            pid_data['date_dif'] = pid_data['timepoint_days']
        pid_data.sort_values('date_dif', inplace=True)
        treatment_days = pid_data.loc[~pid_data['drug'].isnull(), 'date_dif'].tolist()
        if len(treatment_days) == 0:
            no_tday.append(pid_data)
        elif len(treatment_days) == 1:
            one_tday.append(pid_data)
        elif len(treatment_days) == 2:
            two_tday.append(pid_data)
        else:
            print("This shouldn't happen...")
            raise ValueError
    no_tday = pd.concat(no_tday)
    one_tday = pd.concat(one_tday)
    two_tday = pd.concat(two_tday)
    # The way the algorithm is set up, it runs sort of separately on each set of treatment day numbers
    # First, we run on patient ids with no treatment days
    classed_no_tday = []
    # Loop over all patient ids with no treatment days
    for pid in no_tday['participant_id'].unique():
        # Subset to that specific patient id
        pid_data = no_tday.loc[no_tday['participant_id'] == pid]
        # Sort the dataframe based on date_dif date column
        pid_data.sort_values('date_dif', inplace=True)
        study_data = pid_data.loc[~pid_data['HRP2_pg_ml'].isnull()]
        study_days = study_data['date_dif'].tolist()
        study_days.sort()
        if (study_data['HRP2_pg_ml'] < 1.4).all():
            pid_data.loc[pid_data['date_dif'].isin(study_days), 'class'] = 'blue'
        else:
            infection = 0
            for day_i in range(0, len(study_days)):
                day = study_days[day_i]
                day_data = pid_data.loc[pid_data['date_dif'] == day]
                try:
                    day_hrp2 = day_data['HRP2_pg_ml'].item()
                except ValueError:
                    print(pid, day)
                    pass
                # classify all points
                if day_hrp2 < 1.4 and infection == 0:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'blue'
                elif day_hrp2 >= 1.4:
                    if day_hrp2 > 2:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    else:
                        try:
                            prev_day = study_days[day_i - 1]
                            prev_data = pid_data.loc[pid_data['date_dif'] == prev_day]
                            prev_class = prev_data['class'].item()
                            if prev_class in ['blue', 'purple']:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                            else:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = prev_class
                        except IndexError:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    infection = 1
                elif day_hrp2 < 1.4 and infection == 1:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
        classed_no_tday.append(pid_data)
    classed_no_tday = pd.concat(classed_no_tday)

    classed_one_tday = []
    for pid in one_tday['participant_id'].unique():
        pid_data = one_tday.loc[one_tday['participant_id'] == pid]
        pid_data.sort_values('date_dif', inplace=True)
        study_days = pid_data.loc[~pid_data['HRP2_pg_ml'].isnull(), 'date_dif'].tolist()
        study_days.sort()
        tday = pid_data.loc[~pid_data['drug'].isnull(), 'date_dif'].item()
        infection = 0
        for day_i in range(0, len(study_days)):
            day = study_days[day_i]
            day_data = pid_data.loc[pid_data['date_dif'] == day]
            try:
                day_hrp2 = day_data['HRP2_pg_ml'].item()
            except ValueError:
                print(pid, day)
                pass
            # classify symptomatic points
            if (tday - 2) <= day <= (tday + 2):
                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'red'
            # classify points before the treatment
            elif day < tday:
                if day_hrp2 < 1.4 and infection == 0:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'blue'
                elif day_hrp2 >= 1.4:
                    if day_hrp2 > 2:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    else:
                        try:
                            prev_day = study_days[day_i - 1]
                            prev_data = pid_data.loc[pid_data['date_dif'] == prev_day]
                            prev_class = prev_data['class'].item()
                            if prev_class in ['blue', 'purple']:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                            else:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = prev_class
                        except IndexError:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    infection = 1
                elif day_hrp2 < 1.4 and infection == 1:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
            # classify points after the treatment
            elif day > tday:
                if (tday + 20) < day:
                    if day_hrp2 > 4:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    elif day_hrp2 < 1.4:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                    else:
                        try:
                            prev_day = study_days[day_i - 1]
                            if prev_day > tday:
                                prev_data = pid_data.loc[pid_data['date_dif'] == prev_day]
                                prev_class = prev_data['class'].item()
                                prev_hrp2 = prev_data['HRP2_pg_ml'].item()
                                if prev_class != 'red':
                                    if (day_hrp2 - 1 > prev_hrp2) or (prev_class == 'purple'):
                                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                                    else:
                                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = prev_class
                                else:
                                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                            else:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                        except IndexError:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                else:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
        classed_one_tday.append(pid_data)
    classed_one_tday = pd.concat(classed_one_tday)

    classed_two_tday = []
    for pid in two_tday['participant_id'].unique():
        pid_data = two_tday.loc[two_tday['participant_id'] == pid]
        pid_data.sort_values('date_dif', inplace=True)
        study_days = pid_data.loc[~pid_data['HRP2_pg_ml'].isnull(), 'date_dif'].tolist()
        study_days.sort()
        tdays = pid_data.loc[~pid_data['drug'].isnull(), 'date_dif'].tolist()
        tdays.sort()
        infection = 0
        for day_i in range(0, len(study_days)):
            day = study_days[day_i]
            day_data = pid_data.loc[pid_data['date_dif'] == day]
            try:
                day_hrp2 = day_data['HRP2_pg_ml'].item()
            except ValueError:
                print(pid, day)
                pass
            # classify symptomatic points
            if any(((tday - 2) <= day <= (tday + 2)) for tday in tdays):
                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'red'
            # classify points before any treatment
            elif all(day < tday for tday in tdays):
                if day_hrp2 < 1.4 and infection == 0:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'blue'
                elif day_hrp2 >= 1.4:
                    if day_hrp2 > 2:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    else:
                        try:
                            prev_day = study_days[day_i - 1]
                            prev_data = pid_data.loc[pid_data['date_dif'] == prev_day]
                            prev_class = prev_data['class'].item()
                            if prev_class in ['blue', 'purple']:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                            else:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = prev_class
                        except IndexError:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    infection = 1
                elif day_hrp2 < 1.4 and infection == 1:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
            # classify points after any treatment
            elif all(day > tday for tday in tdays):
                if (tday + 20) < day:
                    if day_hrp2 > 4:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                    elif day_hrp2 < 1.4:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                    else:
                        try:
                            prev_day = study_days[day_i - 1]
                            if all(prev_day > tday for tday in tdays):
                                prev_data = pid_data.loc[pid_data['date_dif'] == prev_day]
                                prev_class = prev_data['class'].item()
                                prev_hrp2 = prev_data['HRP2_pg_ml'].item()
                                if prev_class != 'red':
                                    if (day_hrp2 - 1 > prev_hrp2) or (prev_class == 'purple'):
                                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                                    else:
                                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = prev_class
                                else:
                                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                            else:
                                pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                        except IndexError:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                else:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
            # classify points in between treatments
            else:
                tday_range = tdays[-1] - tdays[0]
                tday_mid = sum(tdays) / 2
                tday_start = tdays[0]
                tday_end = tdays[-1]
                if day_hrp2 < 1.4:
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                elif day < (tday_start + (tday_range / 3)):
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                elif day > (tday_end - (tday_range / 3)):
                    pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                else:
                    try:
                        prev_day = study_days[day_i - 1]
                        prev_hrp2 = pid_data.loc[pid_data['date_dif'] == prev_day, 'HRP2_pg_ml'].item()
                        prev_class = pid_data.loc[pid_data['date_dif'] == prev_day, 'class'].item()
                        if day_hrp2 > prev_hrp2:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'yellow'
                        else:
                            pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'green'
                    except IndexError:
                        pid_data.loc[pid_data['date_dif'] == day, 'class'] = 'I HAVE NO IDEA, DOES THIS EVER HAPPEN?'
        classed_two_tday.append(pid_data)
    classed_two_tday = pd.concat(classed_two_tday)

    classed_data = pd.concat([classed_no_tday, classed_one_tday, classed_two_tday])
    classed_data.sort_values('participant_id', inplace=True)

    split_pdf = PdfPages('C:/Users/lzoeckler/Desktop/mali_meta/NIH_mali_class_attempt_day_split.pdf')
    for pid in classed_data['participant_id'].unique():
        combo = classed_data.loc[classed_data['participant_id'] == pid]
        if len(combo) > 0:
            combo.sort_values('date_dif', inplace=True)
            for_line = combo.loc[~combo['HRP2_pg_ml'].isnull()]
            max_day = max(combo['date_dif'])
            min_day = min(combo['date_dif'])
            try:
                max_y = max([max(for_line['HRP2_pg_ml']), max(for_line['LDH_Pan_pg_ml'])])
                min_y = min([min(for_line['HRP2_pg_ml']), min(for_line['LDH_Pan_pg_ml'])])
            except ValueError:
                print(pid)
                continue
            f, ax1 = plt.subplots()
            age = combo['age_yrs'].unique()[0]
            treatment_days = combo.loc[~combo['drug'].isnull(), 'date_dif'].tolist()
            for day in treatment_days:
                ax1.plot(np.array([day, day]), np.array([0, 6.5]), color='purple',
                         linestyle='--', alpha=0.6)
            hrp2_urdt_lod = ax1.plot(np.array([min_day, max_day]), np.array([1.4, 1.4]),
                                     color='red', linestyle='--', alpha=0.6)
            hrp2_urdt_lod = ax1.plot(np.array([min_day, max_day]), np.array([3, 3]),
                                     color='green', linestyle='--', alpha=0.6)
            title = """patient_id: {}""".format(pid)
            ln4 = ax1.plot(for_line['date_dif'], for_line['HRP2_pg_ml'],
                           c='black', alpha=0.6, label='HRP2')
            ln3 = ax1.plot(for_line['date_dif'], for_line['LDH_Pan_pg_ml'],
                           c='green', alpha=0.6, label='pLDH')
            try:
                ax1.scatter(for_line['date_dif'], for_line['HRP2_pg_ml'], c=for_line['class'])
            except ValueError:
                print(pid)
                continue
            ax1.set_title(title)
            ax1.set_xlabel('Time point, in days')
            ax1.set_ylabel('Log10 of pg/ml')
            ax1.set_ylim([-.5, 7])

            # LINE STUFF
            lns = ln4 + ln3
            lns = lns + [Line2D([0], [0], marker='o', color='k', label='Symptomatic', markerfacecolor='r',
                                markersize=10, alpha=0.6)]
            lns = lns + [Line2D([0], [0], marker='o', color='k', label='Clearing', markerfacecolor='g',
                                markersize=10, alpha=0.6)]
            lns = lns + [Line2D([0], [0], marker='o', color='k', label='Chronic', markerfacecolor='y',
                                markersize=10, alpha=0.6)]
            lns = lns + [Line2D([0], [0], marker='o', color='k', label='Discrepancy', markerfacecolor='k',
                                markersize=10, alpha=0.6)]
            lns = lns + [Line2D([0], [0], marker='o', color='k', label='Uninfected', markerfacecolor='b',
                                markersize=10, alpha=0.6)]
            lns = lns + [Line2D([0], [0], color='red', linestyle='--', label='HRP2 uRDT LoD', alpha=0.6)]
            lns = lns + [Line2D([0], [0], color='green', linestyle='--', label='Target pLDH uRDT LoD', alpha=0.6)]
            lns = lns + [Line2D([0], [0], color='purple', linestyle='--', label='Treated', alpha=0.6)]
            labs = [line.get_label() for line in lns]
            ax1.legend(lns, labs, bbox_to_anchor=(.75, -.2), ncol=2)

            # Actually plot stuff
            plt.tight_layout()
            split_pdf.savefig(f)
            plt.show()
            plt.close()
        else:
            print(pid)
    split_pdf.close()


if __name__ == "__main__":
    'C:/Users/lzoeckler/Desktop/mali_meta'
    main()