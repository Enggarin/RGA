#Graph preparation
import pandas as pd

stage = 1

if stage == 1:
	#Stage 1
	r_net = pd.read_csv('road_net.csv', delimiter=';')

	##Add other side of road graph (backward)
	r_net['direction'] = 1
	back_net = r_net[r_net.one_way==0]
	back_net.rename({'source': 't', 'target':'s'}, axis=1, inplace=True)
	back_net.rename({'t': 'target', 's':'source'}, axis=1, inplace=True)
	back_net['direction'] = -1
	r_net = pd.concat([r_net, back_net], axis=0, ignore_index=True, sort=False)

	##Remove loops - you should check ur graph before!
	r_net['s_e'] = r_net.source.astype(str) + '_' + r_net.target.astype(str)
	r_net.sort_values(by=['s_e', 'length'], inplace = True)
	r_net.drop_duplicates(subset ='s_e', keep = 'first', inplace = True)
	r_net.drop(['s_e'], axis=1, inplace=True)

	r_net.to_csv('r_net_s1.csv', sep=';')

elif stage == 2:
	#Stage 2
	##Find and mark adjoining edges
	r_net = pd.read_csv('r_net_s1.csv', delimiter=';')
	for t in r_net.itertuples():
	    temp_df = r_net[r_net.source == t.target]
	    for nei in temp_df.itertuples(): #TODO add more then 1 side node
	        if temp_df.shape[0] > 1:
	            D = nei.degree - t.degree
	            if (D < -20 and D > -160) or (D < 340 and D > 200):
	                r_net.loc[t.Index, 'left_node'] = nei.target
	            elif (D > 20 and D < 160) or (D > -340 and D < -200):
	                r_net.loc[t.Index, 'right_node'] = nei.target
	r_net.fillna(value=0, inplace=True)
	r_net['left_node'] = r_net['left_node'].astype('int32')
	r_net['right_node'] = r_net['right_node'].astype('int32')

	r_net.to_csv('r_net_adv.csv', sep=';')
