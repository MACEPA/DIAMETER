import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
pd.set_option('chained_assignment', None)


# This function adjusts the treatment_condition temperatures to integer values
def temp_adjust(df):
    fill = df['treatment_condition']
    fill = fill.replace("oC", "")
    fill = int(fill)
    return fill


def main(input_dir):
    # Read in the DBS stability data
    main_df = pd.read_excel('{}/DBS_stability_2.xlsx'.format(input_dir))
    # Rename columns
    main_df.rename(columns={'Unnamed: 0': 'analyte', 'Unnamed: 1': 'treatment_condition',
                            'Unnamed: 2': 'time_or_humidity'}, inplace=True)
    # Rename some analyte values
    main_df.loc[main_df['analyte'] == 'Pv LDH', 'analyte'] = 'Pv_LDH'
    main_df.loc[main_df['analyte'] == 'Pf LDH', 'analyte'] = 'Pf_LDH'
    # subset to just time treatment condition, not humidity
    time_df = main_df.loc[main_df['treatment_condition'] != 'Humidity']
    time_df['treatment_condition'] = time_df.apply(temp_adjust, axis=1)
    # Create a subset dataframe for the base temp, -20
    base_value_df = time_df.loc[time_df['treatment_condition'] == -20]
    base_value_df = base_value_df.groupby(['analyte', 'treatment_condition']).mean().reset_index()
    # Drop unnecessary columns in the subset base dataframe
    base_value_df.drop(['treatment_condition', 'time_or_humidity'], axis=1, inplace=True)
    # Subset the rest of the data to only include non base temp rows
    time_df = time_df.loc[time_df['treatment_condition'] != -20]
    # Merge on the base dataframe with a '_base' suffix
    time_df = time_df.merge(base_value_df, on='analyte', suffixes=('', '_base'))
    # Create a subset dataframe for the final time point, 240
    end_df = time_df.loc[time_df['time_or_humidity'] == 240.0]
    # Drop unnecessary columns in the subset final dataframe
    end_df.drop('time_or_humidity', axis=1, inplace=True)
    # Set a decay column to "true" (1)
    end_df['decay'] = 1
    # Loop through each of the three concentration values
    for concentration in ['Hi', 'Med', 'Low']:
        # Set the decay column equal to 0 where the final concentration value is greater
        # than the base concentration value (didn't decay)
        end_df.loc[end_df[concentration] > end_df['{}_base'.format(concentration)], 'decay'] = 0
        # Drop the concentration and base concentration columns, keeping decay column
        end_df.drop([concentration, '{}_base'.format(concentration)], axis=1, inplace=True)
    # Merge on the end dataframe, which adds the decay column
    time_df = time_df.merge(end_df, on=['analyte', 'treatment_condition'])
    # Subset to only values with a decay of 1 (rows that decayed)
    time_df = time_df.loc[time_df['decay'] == 1]
    # Drop the decay column after subsetting
    time_df.drop('decay', axis=1, inplace=True)
    # Divide the base values by two
    time_df[['Hi_base', 'Med_base', 'Low_base']] = time_df[['Hi_base', 'Med_base', 'Low_base']].divide(2)
    # Log the data columns
    cols = ['Hi', 'Med', 'Low', 'Hi_base', 'Med_base', 'Low_base']
    time_df[cols] = time_df[cols].applymap(math.log)
    # Loop through each analyte, adding plots to a PDF
    main_dfs = []
    pp = PdfPages('{}/log_regression_log_space.pdf'.format(input_dir))
    for analyte in time_df['analyte'].unique():
        # Subset to just the data associated with the given analyte
        an_df = time_df.loc[time_df['analyte'] == analyte]
        # Loop through each concentration
        analyte_dfs = []
        for concentration in ['Hi', 'Med', 'Low']:
            # Subset to just the given concentration and specific columns
            con_df = an_df[['analyte', 'treatment_condition', 'time_or_humidity',
                            concentration, '{}_base'.format(concentration)]]
            # Pull out the half base value
            half_base = con_df['{}_base'.format(concentration)].tolist()[0]
            # Loop over each unique temperature
            concentration_dfs = []
            for temp in con_df['treatment_condition'].unique():
                # Subset to just the given temperature
                temp_df = con_df.loc[con_df['treatment_condition'] == temp]
                # Drop the half base value
                temp_df.drop('{}_base'.format(concentration), axis=1, inplace=True)
                # Fit a linear regression on the time and concentration value
                regr = linear_model.LinearRegression()
                time = temp_df['time_or_humidity'].values.reshape(-1, 1)
                con = temp_df[concentration].values.reshape(-1, 1)
                regr.fit(time, con)
                # Predict time values
                pred = regr.predict(time)
                # Get the R2 score
                score = r2_score(con, pred)
                # Get the coefficient and intercept
                coef = np.float(regr.coef_)
                intercept = np.float(regr.intercept_)
                # Fine the logged half life of the value, using the coefficient and intercept
                log_half_life = (half_base - intercept) / coef
                # If the half life is negative, set it to null
                if log_half_life < 0:
                    log_half_life = np.nan
                # Otherwise, round it to three decimal places
                else:
                    log_half_life = round(log_half_life, 3)
                f = plt.figure()
                # Scatter plot time and concentration
                plt.scatter(time, con)
                # Line plot time and prediction, in red
                plt.plot(time, pred, color='red')
                # Create a plot title with lots of different information including:
                # the analyte, concentration, temperature, half base value, logged half life,
                # coefficient, intercept, and R2 score
                title = """analyte: {}, concentration: {}, temp: {}, \nhalf life amount: {}, 
                half life time estimate: {}, \nslope: {}, intercept: {}, \nR2: {}""".format(
                    analyte, concentration, temp, round(half_base, 3), log_half_life, round(coef, 8),
                    round(intercept, 8), score)
                plt.title(title)
                plt.tight_layout()
                pp.savefig(f)
                # Everything after this point is just recombining data elements, in case we want to
                # output a dataframe at some point in the future
                temp_df['{}_half_life'.format(concentration)] = log_half_life
                concentration_dfs.append(temp_df)
            concentration_df = pd.concat(concentration_dfs)
            concentration_df.drop(concentration, axis=1, inplace=True)
            analyte_dfs.append(concentration_df)
        analyte_df = reduce(lambda left, right: pd.merge(left, right,
                                                         on=['analyte', 'treatment_condition',
                                                             'time_or_humidity']), analyte_dfs)
        main_dfs.append(analyte_df)
    # Except for this, this closes the PDF
    pp.close()
    main_df = pd.concat(main_dfs)
    main_df.drop('time_or_humidity', axis=1, inplace=True)
    main_df.drop_duplicates(inplace=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-id', '--input_dir', type=str,
                        default='C:/Users/lzoeckler/Desktop/decay',
                        help='Input directory')
    args = parser.parse_args()
    main(input_dir=args.input_dir)
