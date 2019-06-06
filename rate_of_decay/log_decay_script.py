import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib.backends.backend_pdf import PdfPages
pd.set_option('chained_assignment', None)


def temp_adjust(df):
    fill = df['treatment_condition']
    fill = fill.replace("oC", "")
    fill = int(fill)
    return fill


main_df = pd.read_excel('C:/Users/lzoeckler/Desktop/decay/DBS_stability_2.xlsx')
main_df.rename(columns={'Unnamed: 0': 'analyte', 'Unnamed: 1': 'treatment_condition',
                        'Unnamed: 2': 'time_or_humidity'}, inplace=True)
main_df.loc[main_df['analyte'] == 'Pv LDH', 'analyte'] = 'Pv_LDH'
main_df.loc[main_df['analyte'] == 'Pf LDH', 'analyte'] = 'Pf_LDH'

time_df = main_df.loc[main_df['treatment_condition'] != 'Humidity']
time_df['treatment_condition'] = time_df.apply(temp_adjust, axis=1)
time_df['treatment_condition'].unique()

base_value_df = time_df.loc[time_df['treatment_condition'] == -20]
base_value_df = base_value_df.groupby(['analyte', 'treatment_condition']).mean().reset_index()
base_value_df.drop(['treatment_condition', 'time_or_humidity'], axis=1, inplace=True)

time_df = time_df.loc[time_df['treatment_condition'] != -20]
time_df = time_df.merge(base_value_df, on='analyte', suffixes=('', '_base'))

end_df = time_df.loc[time_df['time_or_humidity'] == 240.0]
end_df.drop('time_or_humidity', axis=1, inplace=True)
end_df['decay'] = 1
for concentration in ['Hi', 'Med', 'Low']:
    end_df.loc[end_df[concentration] > end_df['{}_base'.format(concentration)], 'decay'] = 0
    end_df.drop([concentration, '{}_base'.format(concentration)], axis=1, inplace=True)

time_df = time_df.merge(end_df, on=['analyte', 'treatment_condition'])
time_df = time_df.loc[time_df['decay'] == 1]
time_df.drop('decay', axis=1, inplace=True)
time_df[['Hi_base', 'Med_base', 'Low_base']] = time_df[['Hi_base', 'Med_base', 'Low_base']].divide(2)

log_df = time_df.copy(deep=True)
cols = ['Hi', 'Med', 'Low', 'Hi_base', 'Med_base', 'Low_base']
log_df[cols] = log_df[cols].applymap(math.log)

test_df = log_df.copy(deep=True)
main_dfs = []
pp = PdfPages('C:/Users/lzoeckler/Desktop/decay/log_regression_log_space.pdf')
for analyte in time_df['analyte'].unique():
    an_df = test_df.loc[test_df['analyte'] == analyte]
    analyte_dfs = []
    for concentration in ['Hi', 'Med', 'Low']:
        con_df = an_df[['analyte', 'treatment_condition', 'time_or_humidity',
                        concentration, '{}_base'.format(concentration)]]
        half_base = con_df['{}_base'.format(concentration)].tolist()[0]
        concentration_dfs = []
        for temp in con_df['treatment_condition'].unique():
            temp_df = con_df.loc[con_df['treatment_condition'] == temp]
            temp_df.drop('{}_base'.format(concentration), axis=1, inplace=True)
            regr = linear_model.LinearRegression()
            time = temp_df['time_or_humidity'].values.reshape(-1, 1)
            con = temp_df[concentration].values.reshape(-1, 1)
            regr.fit(time, con)
            pred = regr.predict(time)
            score = r2_score(con, pred)
            coef = np.float(regr.coef_)
            intercept = np.float(regr.intercept_)
            log_half_life = (half_base - intercept) / coef
            if log_half_life < 0:
                log_half_life = np.nan
            else:
                log_half_life = round(log_half_life, 3)
            trans_con = np.exp(con)
            trans_pred = np.exp(pred)
            f = plt.figure()
            plt.scatter(time, con)
            plt.plot(time, pred, color='red')
            true_half_base = np.exp(half_base)
            title = """analyte: {}, concentration: {}, temp: {}, \nhalf life amount: {}, 
            half life time estimate: {}, \nslope: {}, intercept: {}, \nR2: {}""".format(
                analyte, concentration, temp, round(half_base, 3), log_half_life, round(coef, 8),
                round(intercept, 8), score)
            plt.title(title)
            plt.tight_layout()
            pp.savefig(f)
            temp_df['{}_half_life'.format(concentration)] = log_half_life
            concentration_dfs.append(temp_df)
        concentration_df = pd.concat(concentration_dfs)
        concentration_df.drop(concentration, axis=1, inplace=True)
        analyte_dfs.append(concentration_df)
    analyte_df = reduce(lambda left, right: pd.merge(left, right, on=['analyte', 'treatment_condition',
                                                                      'time_or_humidity']), analyte_dfs)
    main_dfs.append(analyte_df)
pp.close()
main_df = pd.concat(main_dfs)
main_df.drop('time_or_humidity', axis=1, inplace=True)
main_df.drop_duplicates(inplace=True)
