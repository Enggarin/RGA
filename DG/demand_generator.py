# Demo demand generator - just to create demo data for morninig demand
import pandas as pd
import numpy as np
import math
import os


if os.path.exists('DG_OUT') == False:                
    os.mkdir('DG_OUT')
                  
d_spred = pd.read_csv('d_spred_m.csv', sep=';') # relative distribution of morning demand
td_dist = pd.read_csv('td_dist.csv', sep=';') # distances beetwen districts
td_feach = pd.read_csv('td_feach.csv', sep=';') # main features of districts: population, workplaces

dec_type = 'parabolic' # or 'linear' or 'sigmoid'

def lin_dec(x):
    return -0.2*x+1

def par_dec(x):
    return -0.15*x**2-0.1*x+1

def sig_dec(x):
    return -1/(1.85+math.e**(-(10*x-5)))+1

if dec_type == 'linear':
    td_dist['dec_k'] = lin_dec((td_dist.dist-td_dist.dist.min())/(td_dist.dist.max()-td_dist.dist.min()))
elif dec_type == 'parabolic':
    td_dist['dec_k'] = par_dec((td_dist.dist-td_dist.dist.min())/(td_dist.dist.max()-td_dist.dist.min()))
elif dec_type == 'sigmoid':
    td_dist['dec_k'] = sig_dec((td_dist.dist-td_dist.dist.min())/(td_dist.dist.max()-td_dist.dist.min()))

td_ar = list(td_feach.td.values)
df_destricts = [f'd{i}' for i in td_ar]
trip_distr = pd.DataFrame(data=None, index=df_destricts, columns=df_destricts)

for per in d_spred.itertuples():
    # First fill trips table
    for i in range(len(td_ar)):
        for j in range(len(td_ar)):
            if i == j:
                dec_k = 1 # push 0 if u want zero for internal district trips
            else:
                dec_k = float(td_dist[(td_dist.from_td==i+1) & (td_dist.to_td==j+1)]['dec_k'])
            trip_distr.iloc[i, j] = int(dec_k*td_feach.wp_q[j]*per.k_irreg*td_feach.pop_q[i]/td_feach.pop_q.sum())
    # K correction coef
    trip_cols = trip_distr.columns
    k_cor = []
    for i in range(len(trip_cols)):
        k_cor.append(td_feach.wp_q[i]*per.k_irreg/trip_distr[trip_cols[i]].sum())
    # Fin correction
    for i in range(len(trip_cols)):
        trip_distr[trip_cols[i]] = np.round(trip_distr[trip_cols[i]].values*k_cor[i], 0)
        trip_distr.to_csv(f'DG_OUT/trip_distr{per.time_h}.csv', sep=';')
    for s in trip_distr.itertuples():
        for v in trip_distr.columns:
            idx = td_dist.index[(td_dist['from_td'] == int(s.Index[1:])) & (td_dist['to_td'] == int(v[1:]))].values
            td_dist.loc[idx[0], str(f'h_{per.time_h}')] = trip_distr[s.Index][v]
            
td_dist.to_csv('DG_OUT/trip_distr_all_time.csv', sep=';', index=False)


