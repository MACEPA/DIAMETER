# function for cleaning concentration info
def fix_concentrations(df):
    con = df['concentration'].partition(':')[2]
    con = con.partition(')')[0]
    if len(con) != 0:
        return con
    else:
        return '1'


# generate dilution constants based on initial dilution value
def build_dil_constants(base_dil):
    return {str(base_dil**i): (base_dil**(i-1)) for i in range(1, 10)


# generate dilution sets based on initial dilution value
def build_dil_sets(base_dil):
    return {str(base_dil ** i): (str(base_dil ** (i - 1)), str(base_dil ** 1), 'fail') for i in range(1, 10)}


# threshhold values for various analytes
THRESHOLDS = {4: {'ulq': {'HRP2_pg_ml': 330, 'LDH_Pan_pg_ml': 10514, 'LDH_Pv_pg_ml': 497, 'CRP_ng_ml': 9574},
                  'llq': {}},
              5: {'ulq': {'HRP2_pg_ml': 2800, 'LDH_Pan_pg_ml': 67000, 'LDH_Pv_pg_ml': 19200, 'LDH_Pf_pg_ml': 20800,
                          'CRP_ng_ml': 38000},
                  'llq': {'HRP2_pg_ml': .68, 'LDH_Pan_pg_ml': 16.36, 'LDH_Pv_pg_ml': 4.96, 'LDH_Pf_pg_ml': 5.08,
                          'CRP_ng_ml': 9.28}}}
