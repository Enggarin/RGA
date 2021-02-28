#RGA v0.1
#GNU General Public License
import pandas as pd
import numpy as np
from collections import deque
import itertools
import math
import json
import os

if os.path.exists('LOG') == False:                
    os.mkdir('LOG')

# Some globals
batch = 5/60 # >> The batch of flow movement time, hours (proper values for numerator: 1, 2, 3, 4, 5)
nodes_trsh = 20 # >> Take a part of nodes for each district (for better perfomance take more then 10) = quantity of nodes in district/nodes_trsh
# while calculating flow distribution, 1/nodes_trsh
num_periods = int(1/batch)
max_q = 145 # The max density for 1 line (vehicles/km)
search_limit = 10000 # limit for nodes in path (pathfinder func)

lines_penalty = { # Set speed penalty for multilines roads, TODO improve
    1: 1,
    2: 0.98,
    3: 0.96,
    4: 0.93,
    5: 0.9,
    6: 0.84
}

satur_flow = { #Saturation flow depend of line maneuver/ Поток насыщения полосы движения в зависимости от разрешеных маневров
    1: 1800,
    2: 1600,
    3: 1500,
    4: 1600,
    5: 1500,
    6: 1600
}

### INFO about type of segments/edge of graph (r_net.type_seg) ###
## 1, 2, 3 - простые классы, используется если граф не размечен по типу дорог и пересечений
# 1 - main - speed road / магистраль
# 2 - main / перегон, главная дорога
# 3 - secondary / перегон, второстепенная дорога 
# 4 - controlled intersection / регулируемое пересечение
# 5 - uncontrolled intersection on main / нерегулируемое пересечение на главной дороге
# 6 - uncontrolled intersection on secondary / нерегулируемое пересечение на второстепенной дороге
# 7 - round intersection / круговое пересечение

### line maneuver / разрешенные направления движения по полосе ####
# 1 - srtight
# 2 - left turns
# 3 - right turns
# 4 - left and strigth
# 5 - stright and right
# 6 - left, stright, right

### Maneuver direction (mvn_dir) ###
#1 - stright
#2 - left
#3 - right

str_ar = [1, 4, 5, 6] # link beetwen line maneuver and mvn_dir
lft_ar = [2, 4, 6]
rgt_ar = [3, 5, 6]

# Load the matrix of demand (flow values beetwen each other transport districts)
# Note > previously delete all rows with zero total trips and after load it
trip_matrix = pd.read_csv('trip_matrix.csv', delimiter=';')
nodes_td = pd.read_csv('nodes-districts.csv', delimiter=';', dtype = {'id_node': np.int16, 'id_td': np.int16})

# # Load information about exist the road traffic scheme (restrictions on maneuvers)
restricts = pd.read_csv('restricts.csv', delimiter=';') 
cc_dict = json.load(open('cc_description.json'))

# The main road description
r_net = pd.read_csv('r_net_adv.csv', delimiter=';')

# Init base point of graph state - suitabel for morning: density, speed, pass time
r_net['q_0'] = 0
r_net['v_0'] = 0
free_net_q = 5

for i in r_net.itertuples():
    if i.type_seg == 1:
        r_net.loc[i.Index, 'v_0'] = round(153 - 29.5*np.log(free_net_q), 2)
    elif i.type_seg == 2:
        r_net.loc[i.Index, 'v_0'] = round(72 - 13*np.log(free_net_q), 2)
    elif i.type_seg == 3:
        r_net.loc[i.Index, 'v_0'] = round(61 - 10.82*np.log(free_net_q), 2)
    elif i.type_seg == 4: 
        r_net.loc[i.Index, 'v_0'] = round(61 - 10.82*np.log(free_net_q), 2) # TODO improve it
    elif i.type_seg == 5:
        r_net.loc[i.Index, 'v_0'] = round(61 - 10.82*np.log(free_net_q), 2) # TODO improve it
    elif i.type_seg == 6:
        r_net.loc[i.Index, 'v_0'] = round(61 - 10.82*np.log(free_net_q), 2) # TODO improve it
    elif i.type_seg == 7:
        r_net.loc[i.Index, 'v_0'] = round(61 - 10.82*np.log(free_net_q), 2) # TODO improve it
r_net['t_0'] = round(r_net['length']/r_net['v_0'], 5)

# Premade matrix_table with additional columns - periods relatively chosen bath size 
for h in trip_matrix.columns[2:]: 
    trip_matrix[h] = trip_matrix[h].apply(lambda x: np.random.poisson(x/num_periods, num_periods))

start_hour = trip_matrix.columns[2]  # start time point / это для сохранения точки отсчета
start_hour = int(start_hour[1:])
total_h = trip_matrix.columns[2:]

periods = [p for p in range(int(len(total_h)/batch))]

# Create the dict with trips beetwen reduced nodes by periods
tdf = {}
count = 0

for i in trip_matrix.itertuples():
    nodes_out = nodes_td[nodes_td.id_td == i.from_td]['id_node']
    nodes_in = nodes_td[nodes_td.id_td == i.to_td]['id_node']
    if len(nodes_out) == 1:
        t1 = 0
    else:
        t1 = np.random.randint(0, high=len(nodes_out)-1, size=int(np.ceil(len(nodes_out)/nodes_trsh)))
    if len(nodes_in) == 1:
        t2 = 0
    else:       
        t2 = np.random.randint(0, high=len(nodes_in)-1, size=int(np.ceil(len(nodes_in)/nodes_trsh)))
        
    t1 = np.unique(t1)
    t2 = np.unique(t2)
    nodes_out = [nodes_out.values[i] for i in t1]
    nodes_in = [nodes_in.values[i] for i in t2]
    
    if len(nodes_in) >= 1 and len(nodes_out) >= 1: # TODO plan, make cheking before
        current_period = 0
        for hour in trip_matrix.columns[2:]:
            period_arr = eval('i.' + hour)
            for idx in range(len(period_arr)):
                current_period += 1
                sum_spend_trips = 0
                if len(nodes_out) >= len(nodes_in):
                    for j in nodes_out:
                        if sum_spend_trips < period_arr[idx]:
                            if len(nodes_in) > 1:
                                temp_f = np.random.randint(0, high=len(nodes_in), size=1)
                                target_in_node = nodes_in[temp_f[0]]
                            else:
                                target_in_node = nodes_in[0]
                            current_trips = np.ceil(period_arr[idx]/len(nodes_out))
                            sum_spend_trips += current_trips
                            tdf[count] = {'source': int(j), 'target': int(target_in_node), 
                                                       current_period: current_trips}#, 'td': [i.from_td, i.to_td]}
                            count += 1        
                        else:
                            break
                else:
                    for j in nodes_in:
                        if sum_spend_trips < period_arr[idx]:
                            if len(nodes_out) > 1:
                                temp_f = np.random.randint(0, high=len(nodes_out)-1, size=1)
                                target_out_node = nodes_out[temp_f[0]]
                            else:
                                target_out_node = nodes_out[0]
                            current_trips = np.ceil(period_arr[idx]/len(nodes_in))
                            sum_spend_trips += current_trips
                            tdf[count] = {'source': int(target_out_node), 'target': int(j), 
                                          current_period: current_trips}#, 'td': [i.from_td, i.to_td]}
                            count += 1
                        else:
                            break

# Create dict to hold possition of batches while moving through paths
trip_movplace = {}
links_dict = {}
ld_count = 1

for k in tdf:
    for key, value in tdf[k].items():
        if key == 'source':
            source = value
        elif key == 'target':
            target = value
        else:
            first_key  = key # period
            flow = value
    second_key = str(source) + '_' + str(target)
    if second_key not in links_dict:
        links_dict[ld_count] = second_key
        ld_count += 1
    
    if flow == 0:
        continue
    if first_key in trip_movplace:
        trip_movplace[first_key].update({ld_count: {'source': source, 'target': target, 'flow': flow, 'start_t': 0, 'finish_t': 0, 
                                                      'position': [], 'seg_distrib': [], 'state': 0}})
    else:
        trip_movplace[first_key] = {ld_count: {'source': source, 'target': target, 'flow': flow, 'start_t': 0, 'finish_t': 0, 
                                                 'position': [], 'seg_distrib': [], 'state': 0}}

# SKIP by default
# That's way to create dataframe with the trip distribution (if u need it)
# trip_distrib = pd.DataFrame(tdf)
# trip_distrib = trip_distrib.T

# trip_distrib['source'] = trip_distrib['source'].astype('int32')
# trip_distrib['target'] = trip_distrib['target'].astype('int32')
# trip_distrib['s_e'] = trip_distrib.source.astype(str) + '_' + trip_distrib.target.astype(str)
# trip_distrib = trip_distrib.fillna(0)
# trip_distrib.drop(columns=['source', 'target'], inplace=True)
# trip_distrib = trip_distrib.groupby(['s_e']).sum()

# trip_distrib.reset_index(inplace=True)
# trip_distrib['source'] = trip_distrib['s_e'].apply(lambda x: x.split('_')[0])
# trip_distrib['target'] = trip_distrib['s_e'].apply(lambda x: x.split('_')[1])
# trip_distrib['source'] = trip_distrib['source'].astype(float).astype(int)
# trip_distrib['target'] = trip_distrib['target'].astype(float).astype(int)

# trip_distrib_hours = trip_distrib[sorted(trip_distrib.drop(['s_e', 'source', 'target'], axis=1))]
# trip_distrib = pd.concat([trip_distrib[['s_e', 'source', 'target']], trip_distrib_hours], axis=1)

del trip_matrix, nodes_td, tdf, links_dict

d_restricts = {}
for i in restricts.itertuples():
    if i.st_seg in d_restricts:
        d_restricts[i.st_seg].update({i.end_seg: i.to_node})
    else:
        d_restricts[i.st_seg] = {i.end_seg: i.to_node}
del restricts

def add_edge(G, source, target, weight):
    if source not in G:
        G[source] = {target: weight}
    else:
        G[source][target] = weight

def get_G(period):
    G = {}
    for i in r_net.itertuples():               
        pass_time = round(eval('i.t_' + str(period)), 5)
        if pass_time == 0:
            pass_time = round(i.t_0, 5)
        add_edge(G, i.source, i.target, pass_time)
    return G

def deijkstra(G, start):
    try:
        Q = deque()
        F = deque()
        S = {}
        S[start] = 0
        Q.append(start)
        F.append(start)
        L = {}
        while Q:
            v = Q.pop()
            f = F.pop()
            for i in G[v]:
                if f in d_restricts and f != start:
                    if v in d_restricts[f]:
                        if i == d_restricts[f][v]: continue

                if i not in S or S[v] + G[v][i] < S[i]:
                    S[i] = round(S[v] + G[v][i], 5)
                    L[i] = v
                    Q.append(i)
                    F.append(v)
        return L #S, L
    except KeyError:
#         print('func deijkstra - broken graph, start:', start)
        return {}

def pathfinder(links, start, end): # return nodes sequences
    try:
        path = deque()
        path.append(end)
        cur_v = end
        count_br = 0
        while cur_v != start:
            path.appendleft(links[cur_v])
            cur_v = links[cur_v]
            count_br += 1
            if count_br == search_limit: # max nodes in path
#                 print('func pathfinder - broken graph, start, end:', start, end)
                break
        if len(path) > 1:
            return path
        else:
            return []
    
    except KeyError:
#         print('func pathfinder - broken graph, start, end:', start, end)
        return []

def q_converter(q, seg_type, seg_len, seg_numlns, cc_inf=0, mnv_dir=0, flow=0, cur_per=0, majorflow=0, from_node=0):  #seg_flow_rate --> flow
    '''
    Calc speed and pass time on segment by func(by density 'q' and description of intersection)
    Функция расчета скорости и времени в пути
    '''
    if q < 5:
        q = 5
    if seg_type == 1: # main (speed road)
        calc_v = 153 - 29.5*np.log(q)
        calc_v *= lines_penalty[seg_numlns] # todo improve
    elif seg_type == 2: # main road
        calc_v = 72 - 13*np.log(q)
        calc_v *= lines_penalty[seg_numlns]
    elif seg_type == 3 : # secondary road
        calc_v = 61 - 10.82*np.log(q)
        calc_v *= lines_penalty[seg_numlns]
    elif seg_type == 5 : # todo improve
        calc_v = 61 - 10.82*np.log(q)
        calc_v *= lines_penalty[seg_numlns]   
    elif seg_type == 4: # controlled intersection (ref. HCM 2000)
        if cur_per != cc_inf['cur_per']: # обнуляем поток в сегменте - новый период - новый поток накоплением
            cc_inf['cur_per'] = cur_per
            for reset_flow_dir in cc_inf:
                if reset_flow_dir != 'C' and reset_flow_dir != 'cur_per':
                    for reset_flow_ln in cc_inf[reset_flow_dir]:
                        cc_inf[reset_flow_dir][reset_flow_ln][2] = 0

        C = cc_inf['C'] # cycle length (h) / продолжительность цикла
        ln_keys = [] # num lines / номера полос, соответсвующих требуемому направлению движения
        g_lns = [] # green time of lane (h) / зеленая фаза по полосам движения
        type_lns = [] # line types by acc. maneuvers / разрешенные маневры по полосам движения
        for cl in cc_inf[from_node]:
            if mnv_dir == 1: # собираем полосы, по которым может совершить требуемый маневр mvn_dir
                if cc_inf[from_node][cl][0] in str_ar: # and (cl != 'C' and cl != 'cur_per'):
                    ln_keys.append(cl)
                    type_lns.append(cc_inf[from_node][cl][0])
                    g_lns.append(cc_inf[from_node][cl][1])
            elif mnv_dir == 2:
                if cc_inf[from_node][cl][0] in lft_ar:
                    ln_keys.append(cl)
                    type_lns.append(cc_inf[from_node][cl][0])
                    g_lns.append(cc_inf[from_node][cl][1])
#                     flow_cl.append(cl[3])
#                 else:
#                     other_t.append(cl[1])?
            elif mnv_dir == 3:
                if cc_inf[from_node][cl][0] in rgt_ar:
                    ln_keys.append(cl)
                    type_lns.append(cc_inf[from_node][cl][0])
                    g_lns.append(cc_inf[from_node][cl][1])
#                     flow_cl.append(cl[3])
#                 else:
    #                     other_t.append(cl[1])# нужно ли??
        flow = flow/batch # Convert flow value to veh/hour
        flow_spred = np.array([flow*gi/sum(g_lns) for gi in g_lns])          
        k = 0.5 # TODO improve it
        l = 1 # simple mode, TODO improve it
        min_speed = 12.6 # TODO improve it
        min_spd_lns = [min_speed*gi/C for gi in g_lns]
        for ick in range(len(ln_keys)): 
            flow_spred[ick] += int(cc_inf[from_node][ln_keys[ick]][2])
            cc_inf[from_node][ln_keys[ick]][2] = flow_spred[ick]
            v = flow_spred[ick] # flow rate for lane
            g = g_lns[ick] # green time
            s = satur_flow[type_lns[ick]]
            X = (v*C)/(s*g) # capacity ratio
            c = g*s/C # capacity of line (veh/h)
            d1 = (0.5*C*3600*((1-g/C)**2))/(1-abs(min(1, X)*g/C)) # delay 1 (s)
            d2 = 900*batch*abs((X-1)+((X-1)**2+8*k*l*X/(c*batch))**0.5) # delay 2 (s)
            Q = max(0, cc_inf[from_node][ln_keys[ick]][3]+c*batch*(X-1)) # queu (veh)
            Q_km = Q*0.01 # convert vehicles to km
            if Q_km > seg_len: 
                Q_km = seg_len
            cc_inf[from_node][ln_keys[ick]][3] = int(Q)
            if Q == 0:
                d_ud = 0 # duration of unmet demand in T (batch), (h)
            else:
                if X < 1:
                    d_ud = min(batch, Q/(c*(1-X)))
                else:
                    d_ud = batch
            if d_ud < batch:
                u = 0
            else:
                if X < 1:
                    u = 1 - c*batch/(Q*(1-X))
                else:
                    u = 0
            d3 = 1800*Q*(1+u)*d_ud/(c*batch) # delay 3
            if d3 > 0:
                d_sum = d1 + d2 + d3
            else:
                d_sum = d1 + d2
            max_passtime = seg_len/min_spd_lns[ick]
            reach_time = (seg_len - Q_km)/(72 - 13*np.log(q))
            total_time = d_sum/3600 + reach_time
            if total_time > max_passtime:
                total_time = max_passtime
            speed_ln = seg_len/total_time
            cc_inf[from_node][ln_keys[ick]][4] = speed_ln
        speed_lns = np.array([])
        all_lines_flow = np.array([])
        for lns in cc_inf[from_node]:
            speed_lns = np.append(speed_lns, cc_inf[from_node][lns][4])
            all_lines_flow = np.append(all_lines_flow, cc_inf[from_node][lns][2])
        calc_v = sum(speed_lns*all_lines_flow)/sum(all_lines_flow)     
    elif seg_type == 6: # 
        reach_time = seg_len/(72 - 13*np.log(q)) #TODO improve it
        uncc_delay = 1/(1450*math.exp(-0.0007*majorflow)) #hours
        calc_v = seg_len/(reach_time+uncc_delay)
 
    calc_v = round(calc_v, 3)
    t = round(seg_len/calc_v, 5)
    return calc_v, t

# States for trip_movplace (l_state)
# 0 - start first iteration of link
# 1 - end first iter by movement # ???
# 2 - while move (second and next iterations)
# 3 - got finish
# 4 - link done, finish and clear tail

def prot_movement(cur_period, calc_per, link, l_state):
    '''
    Main function - the batch movement on road graph
    '''
    prev_track = trip_movplace[calc_per][link]['position']
    prev_seg_distrib = trip_movplace[calc_per][link]['seg_distrib']
    start_node = trip_movplace[calc_per][link]['source']
    target = trip_movplace[calc_per][link]['target']
    flow = trip_movplace[calc_per][link]['flow']
    print('cur_period, calc_per, link, flow, l_state: ', cur_period, calc_per, link, flow, l_state)
    Graph = get_G(calc_per-1)
    all_paths = deijkstra(Graph, start_node)
    cur_path = pathfinder(all_paths, start_node, target)
    if len(cur_path) == 0:
        print('Graph problems, start node, end_node:', start_node, target, ' Return NONE')
        trip_movplace[calc_per][link]['state'] = 4
        return None 
    
    if l_state == 0:
        cut_index = 0
        trip_movplace[calc_per][link]['start_t'] = start_hour + batch*cur_period - batch/2 # hours!
    else:
        prev_end_node = trip_movplace[calc_per][link]['position'][-1]
        cut_index = cur_path.index(prev_end_node)

    remain_time = batch
    save_tracks = deque()
    save_seg_load = deque() #veh per segment
    sliced_path = deque(itertools.islice(cur_path, cut_index, None))
    
    if l_state != 3:
        #Takes information about segments on path
        for g_key in range(len(sliced_path)-1):
            cur_seg_q = float(r_net[(r_net['source']==sliced_path[g_key]) & 
                                    (r_net['target']==sliced_path[g_key+1])][f'q_{cur_period}'])
            cur_seg_t = float(r_net[(r_net['source']==sliced_path[g_key]) & 
                                  (r_net['target']==sliced_path[g_key+1])][f't_{cur_period}'])
            cur_seg_len = float(r_net[(r_net['source']==sliced_path[g_key]) & 
                                      (r_net['target']==sliced_path[g_key+1])]['length'])
            cur_seg_numlines = int(r_net[(r_net['source']==sliced_path[g_key]) & 
                                   (r_net['target']==sliced_path[g_key+1])]['num_lines'])
            cur_seg_type = int(r_net[(r_net['source']==sliced_path[g_key]) & 
                                     (r_net['target']==sliced_path[g_key+1])]['type_seg'])

            add_seg_load = flow*cur_seg_t/batch
            new_seg_q = round(cur_seg_q + add_seg_load/(cur_seg_len*cur_seg_numlines), 2)
            if new_seg_q <= 145:
                if cur_seg_type == 4:
                    mnv_dir = 1
                    if sliced_path[g_key+1] != cur_path[-1]:
                        if sliced_path[g_key+2] != cur_path[-1]:
                            check_left_turn = r_net[(r_net['source']==sliced_path[g_key]) & 
                                                    (r_net['target']==sliced_path[g_key+1])]['left_node']
                            check_right_turn = r_net[(r_net['source']==sliced_path[g_key]) & 
                                                    (r_net['target']==sliced_path[g_key+1])]['right_node']
                            if sliced_path[g_key+2] == int(check_left_turn):
                                mnv_dir = 2
                            elif sliced_path[g_key+2] == int(check_right_turn):
                                mnv_dir = 3
                    cur_v, cur_t = q_converter(new_seg_q, cur_seg_type, cur_seg_len, cur_seg_numlines, cc_inf=cc_dict[str(sliced_path[g_key+1])], 
                                               mnv_dir=mnv_dir, flow=flow, cur_per=cur_period, from_node=str(sliced_path[g_key]))
                elif cur_seg_type == 6:
                        temp_df_majorflow = r_net[(r_net['target']==sliced_path[g_key+1]) & (r_net['type_seg']!=6)][['q_0', 'v_0']]
                        majorflow = 0
                        if temp_df_majorflow.shape[0] > 0:
                            for mf in temp_df_majorflow.itertuples():
                                majorflow += mf.q_0*mf.v_0
                        cur_v, cur_t = q_converter(new_seg_q, cur_seg_type, cur_seg_len, cur_seg_numlines, majorflow=majorflow)
                else:
                    cur_v, cur_t = q_converter(new_seg_q, cur_seg_type, cur_seg_len, cur_seg_numlines)
            
                if remain_time > 0.75*cur_t:
                    save_tracks.append(sliced_path[g_key])
                    save_seg_load.append(round(add_seg_load, 5))
                    seg_idx = r_net.index[(r_net['source']==sliced_path[g_key]) & (r_net['target']==sliced_path[g_key+1])].values
                    r_net.loc[seg_idx[0], f'q_{cur_period}'] = new_seg_q
                    r_net.loc[seg_idx[0], f'v_{cur_period}'] = cur_v
                    r_net.loc[seg_idx[0], f't_{cur_period}'] = cur_t
                    remain_time -= cur_t
                    
                    if sliced_path[g_key+1] == cur_path[-1]:
                        trip_movplace[calc_per][link]['finish_t'] = round(start_hour + batch*cur_period + cur_t, 5)
                        save_tracks.append(sliced_path[g_key+1])
                        trip_movplace[calc_per][link]['state'] = 3
                        trip_movplace[calc_per][link]['position'] = save_tracks
                        trip_movplace[calc_per][link]['seg_distrib'] = save_seg_load
                        break
                else:
                    if len(save_tracks) > 0:
                        save_tracks.append(sliced_path[g_key])
                        trip_movplace[calc_per][link]['position'] = save_tracks
                        trip_movplace[calc_per][link]['seg_distrib'] = save_seg_load
                        if l_state == 0:
                            l_state = 1
                        elif l_state == 1:
                            l_state = 2
                        trip_movplace[calc_per][link]['state'] = l_state   
                    break
            else:
                if len(save_tracks) > 0:
                    save_tracks.append(sliced_path[g_key])
                    trip_movplace[calc_per][link]['position'] = save_tracks
                    trip_movplace[calc_per][link]['seg_distrib'] = save_seg_load
                    if l_state == 0:
                        l_state = 1
                    elif l_state == 1:
                        l_state = 2
                    trip_movplace[calc_per][link]['state'] = l_state   
                break
    if l_state == 2 and len(save_tracks) > 0 and len(prev_seg_distrib) > 0:
        flow_spred = sum(save_seg_load)
        to_slice = 0
        for i in range(len(prev_seg_distrib)):
            if flow_spred > 0.75*prev_seg_distrib[i]:
                cur_seg_q = float(r_net[(r_net['source']==prev_track[i]) & 
                                            (r_net['target']==prev_track[i+1])][f'q_{cur_period}'])
                cur_seg_len = float(r_net[(r_net['source']==prev_track[i]) & 
                                          (r_net['target']==prev_track[i+1])]['length'])
                cur_seg_numlin = int(r_net[(r_net['source']==prev_track[i]) & 
                                           (r_net['target']==prev_track[i+1])]['num_lines'])
                new_seg_q = round(cur_seg_q - prev_seg_distrib[i]/(cur_seg_len*cur_seg_numlin), 2)
                seg_idx = r_net.index[(r_net['source']==prev_track[i]) & 
                                      (r_net['target']==prev_track[i+1])].values
                r_net.loc[seg_idx[0], f'q_{cur_period}'] = new_seg_q
                flow_spred -= prev_seg_distrib[i]
                to_slice += 1
            else:
                break
        upd_seg_distrib = deque(itertools.islice(prev_seg_distrib, to_slice, None))
        upd_track = deque(itertools.islice(prev_track, to_slice, None))
        trip_movplace[calc_per][link]['position'] = upd_track + deque(itertools.islice(save_tracks, 1, None))
        trip_movplace[calc_per][link]['seg_distrib'] = upd_seg_distrib + save_seg_load
    
    if l_state == 3:
        for i in range(len(prev_seg_distrib)):
            cur_seg_q = float(r_net[(r_net['source']==prev_track[i]) & 
                                        (r_net['target']==prev_track[i+1])][f'q_{cur_period}'])
            cur_seg_len = float(r_net[(r_net['source']==prev_track[i]) & 
                                      (r_net['target']==prev_track[i+1])]['length'])
            cur_seg_numlin = int(r_net[(r_net['source']==prev_track[i]) & 
                                       (r_net['target']==prev_track[i+1])]['num_lines'])
            new_seg_q = round(cur_seg_q - prev_seg_distrib[i]/(cur_seg_len*cur_seg_numlin), 2)
            seg_idx = r_net.index[(r_net['source']==prev_track[i]) & 
                                  (r_net['target']==prev_track[i+1])].values
            r_net.loc[seg_idx[0], f'q_{cur_period}'] = new_seg_q
            trip_movplace[calc_per][link]['state'] = 4
            trip_movplace[calc_per][link]['position'] = []
            trip_movplace[calc_per][link]['seg_distrib'] = []

calc_periods = np.array([])
for cur_period in periods: 
    print('Current period >>> ', cur_period)
    if cur_period == 0: continue
    calc_periods = np.append(calc_periods, cur_period)
    np.random.shuffle(calc_periods)
    calc_periods = calc_periods.astype(int)
    r_net['q_' + str(cur_period)] = r_net['q_' + str(cur_period-1)] 
    r_net['v_' + str(cur_period)] = r_net['v_' + str(cur_period-1)] 
    r_net['t_' + str(cur_period)] = r_net['t_' + str(cur_period-1)] 
    for calc_per in calc_periods:
        links = np.array(list(trip_movplace[calc_per].keys()))
        np.random.shuffle(links)
        period_todel = 1
        print('  >> calculation period: ', calc_per)
        for l in links:
            l_state = trip_movplace[calc_per][l]['state']
            if l_state != 4:
                prot_movement(cur_period, calc_per, l, l_state)
                period_todel = 0
        else:
            if period_todel == 1:
                print(calc_periods, calc_per)
                calc_periods = np.delete(calc_periods, np.where(calc_periods == calc_per))
                print(calc_periods)
    else:
        if cur_period%5 == 0:
            r_net.to_csv(f'LOG/r_net_per{cur_period}.csv', sep=';')

# Export links descriptions with travel time and flow
f = open('links_descr_out.csv', 'w')
f.write('period' + ";" + 'link' + ";" + 'source' + ";" + 'target' + ";" + 'flow' + ";" + 'start time' + ";" + 'finish time' + '\n')
for k in trip_movplace:
    for kk in trip_movplace[k]:
        f.write(str(k) + ";" + str(kk) + ";" + str(trip_movplace[k][kk]['source']) + ";" + str(trip_movplace[k][kk]['target']) + ";" 
                + str(trip_movplace[k][kk]['flow']) + ";" + str(trip_movplace[k][kk]['start_t']) + ";"
                + str(trip_movplace[k][kk]['finish_t']) + ";" + str(trip_movplace[k][kk]['state']) + '\n')
f.close()

