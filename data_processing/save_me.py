good_df = []
bad_df = []
k = 0
for pid in rebuilt_data['patient_id'].unique():
# for pid in ['pa-114', 'pa-020', 'pa-077', 'pa-124', 'pa-129', 'pa-140', 'pa-169']:
    pid_data = rebuilt_data.loc[rebuilt_data['patient_id'] == pid]
    pid_data.sort_values('time_point_days', inplace=True)
    all_times = pid_data['time_point_days'].unique().tolist()
    all_times.sort()
    max_run = 3
    the_rest = []
    i = 0
    end_val = 4
    baddest_section = []
    the_rest_set = set([0])
    while (end_val <= len(all_times)) & (len(the_rest_set) != 0):
        next_start = None
        time_vals = all_times[i:end_val]
#         print('INITIAL VALS', time_vals)
        coef_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
        avg_val = coef_data['HRP2_pg_ml'].mean()
        coef, score = get_coef(coef_data)
        while (coef > -.03) & (len(time_vals) != 1) & (avg_val > 2.5) & (end_val < len(all_times)):
            end_val = end_val + 1
            time_vals = all_times[i:end_val]
            coef_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals)]
            coef, score = get_coef(coef_data)
            avg_val = coef_data['HRP2_pg_ml'].mean()
#             print('FIRST WHILE, coef: {}, times: {}, avg: {}, r2: {}'.format(coef, time_vals, avg_val, score))
            end_data = pid_data.loc[pid_data['time_point_days'].isin(time_vals[-4:])]
            end_coef, end_score = get_coef(end_data)
#             print('TIME VALS', time_vals)
#             print('NORMAL COEF', coef)
#             print('NORMAL SCORE', score)
#             print('END COEF', end_coef)
#             print('END SCORE', end_score)
#             print('AVG_VAL', avg_val)
            if (coef > -.03) & (avg_val > 2.5) & (score < .3) & (end_score < .9):
#                 print('here')
                current_run = len(time_vals)
                next_start = end_val - 1
                if current_run > max_run:
                    max_run = current_run
                    baddest_section = time_vals
#                     print(baddest_section)
#                 print('SECOND WHILE, coef: {}, times: {}, avg: {}, r2: {}'.format(coef, time_vals, avg_val, score))
#             print('-----------------------')
        if next_start:
            i = next_start
        else:
            i = end_val - 3
        try:
            all_times[i + 4]
            end_val = i + 4
        except:
            the_rest = all_times[i:]
            the_rest_set = set(the_rest) - set(time_vals)
            end_val = i + len(the_rest)
    good_vals = pid_data.loc[~pid_data['time_point_days'].isin(baddest_section)]
    good_df.append(good_vals)
    bad_vals = pid_data.loc[pid_data['time_point_days'].isin(baddest_section)]
    bad_df.append(bad_vals)
    if len(bad_vals) > 0:
        print(pid)
        k += 1
good_df = pd.concat(good_df)
bad_df = pd.concat(bad_df)
print(k)