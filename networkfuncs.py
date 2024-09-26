#%%cython
"""
Here we define the base functions, whcih could be Cythonized if needed
"""

import numpy as np
import matplotlib.pyplot as plt
#cimport numpy as np

import random

import itertools
import functools

import copy

import networkx as nx

from util import shallow_flatten, deep_flatten

def nodes_from_couplings(couplings):
    """
    Might discard weights, returns nodes
    """
    return frozenset(shallow_flatten([item[:2] for item in couplings]))

def graph_with_local_chips(graph_nodes, chip_couplings):
    """
    Should be able to take weighted couplings
    """
    chip_g = nx.Graph()
    node_map = {}
    
    chip_size = len(nodes_from_couplings(chip_couplings))
    
    #1st, set up nodes and intra-chip couplings
    for i, node in enumerate(sorted(list(graph_nodes))):
        chip_couplings_shifted = [[q + i*chip_size for q in pair[:2]] + pair[2:] for pair in chip_couplings]
        chip_nodes = nodes_from_couplings(chip_couplings_shifted)
        chip_g.add_nodes_from(chip_nodes)
        chip_g.add_edges_from(chip_couplings_shifted)
        node_map[node] = i
    #Now the new graph has internal chip couplings
    return (chip_g, node_map)

def graph_from_couplings(chip_couplings):
    g = nx.Graph()
    
    g.add_nodes_from(nodes_from_couplings(chip_couplings))
    g.add_edges_from(chip_couplings)
    
    return g

def find_mono_infid(chip_infid,total_qubits,chip_qubits,rate_change):
    mono_infid = (total_qubits-chip_qubits)*rate_change + chip_infid
    return mono_infid

def add_joiners_to_graph(g, chip_g, node_map,
                       chip_joiner_list, chip_size, weight,
                       randomize = False):
    """
    Sets up cross-chip couplings
    
    May take weighted couplings
    """
    
    node_map = dict([node, [i,0]] for node, i in node_map.items())
    
    joiners = chip_joiner_list
    if randomize:
        joiners = random.sample(joiners, len(joiners))
    
    for edge in g.edges:
        l1 = node_map[edge[0]]
        l2 = node_map[edge[1]]
        
        count_1 = l1[1]
        count_2 = l2[1]
        
        joiner_1 = joiners[count_1] + l1[0]*chip_size
        joiner_2 = joiners[count_2] + l2[0]*chip_size
        
        l1[1] = count_1 + 1
        l2[1] = count_2 + 1
        
        if weight is None:
            chip_g.add_edge(joiner_1, joiner_2)
        else:
            chip_g.add_edge(joiner_1, joiner_2, weight=weight)
    return

def put_chips_locally_in_graph(g, chip_couplings, chip_joiner_list, chip_size,
                               joiner_weight = None, randomize=False):
    """
    Takes graph g, puts it on internally coupled chips in straightforward way.
    """
    chip_g, node_map = graph_with_local_chips(g, chip_couplings)
    
    add_joiners_to_graph(g, chip_g, node_map, chip_joiner_list, chip_size, joiner_weight, randomize)
    
    return chip_g

def graph_lambda2(g):
    """
    See https://people.orie.cornell.edu/dpw/orie6334/Fall2016/lecture7.pdf
        in particular, Cheeger's inequality
    """
    return np.amax(np.partition(nx.normalized_laplacian_spectrum(g), 2)[:2])


def best_random_expander(num_chips, degree, num_retries):
    """
    Constructs a 'pretty good' expander via random graphs
    
    Should have degree*num_nodes be even for networkx
    
    To max out performance, optimize without chips in place?
    """
    #random_generator =  np.random.default_rng(1)
    np.random.seed(1)
    random.seed(1)
    
    best_graph = None
    best_lambda2 = 0.0
    
    for i in range(num_retries):
        g = nx.generators.random_graphs.random_regular_graph(degree, num_chips, seed = np.random)
        lambda2 = graph_lambda2(g)
        if lambda2 > best_lambda2:
            best_graph = g
            best_lambda2 = lambda2
            
    return (best_graph, best_lambda2)
    
def best_random_expander_chipped(num_chips, num_retries, num_joiner_retries,
                         chip_coupling_list, chip_joiner_list, joiner_weight = None):
    
    """
    Finds an optimized expander connectivity on chiplets, where each chiplet plays the
    role of a node in the expander.
    Finds a good expander, then finds a good embedding.
    Does so by randomly trying graphs, using statistical likelihood of good expanders
    
    num_chips - how many chiplets (noes in expander)
    num_retries - number of random trials for the initial expander
    num_joiner_retries - number of trandom trials for the embedding
    chip_coupling_list - list of pairs that correspond to couplings
    chip_joiner_list - list of qubit numbers corresponding to where chips would hvave links
    """
    degree = len(chip_joiner_list)
    
    best_graph, best_lambda2 = best_random_expander(num_chips, degree, num_retries)
    
    chip_size = len(nodes_from_couplings(chip_coupling_list))
    
    best_graph_chipped = None
    best_lambda2_chipped = 0.0
    
    for i in range(num_joiner_retries):
        chip_g = put_chips_locally_in_graph(best_graph, chip_coupling_list, chip_joiner_list, chip_size,
                                    joiner_weight = joiner_weight, randomize=True)
        chip_lambda2 = graph_lambda2(chip_g)
        if chip_lambda2 > best_lambda2_chipped:
            best_graph_chipped = chip_g
            best_lambda2_chipped = chip_lambda2
    
    return (best_graph_chipped, best_lambda2_chipped)

def find_worst_path(g, return_paths = False):
    """
    This method returns the worst case or diameter path, as well as the graph diameter
    
    Could try caching by graph, so we don't recompute shortest paths
    For now, returns this info for future re-use
    
    all_pairs_dijkstra(G, cutoff=None, weight='weight')
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra.html#networkx.algorithms.shortest_paths.weighted.all_pairs_dijkstra
    
    all_pairs_shortest_path(G, cutoff=None)
    https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path.html#networkx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path
    
    """
    #if weigted:
    path_weight_by_targ_by_source = dict(nx.algorithms.shortest_paths.weighted.all_pairs_dijkstra(g))
    #else:
    #    nx.algorithms.shortest_paths.unweighted.all_pairs_shortest_path(g)
    
    worst_source_target_path = None
    worst_path_length = 0
    
    for source, (d1, d2) in path_weight_by_targ_by_source.items():
        for (targ, length) in d1.items():
            if length > worst_path_length:
                worst_path_length = length
                worst_source_target_path = [source, targ, d2[targ]]
    
    if return_paths:
        return (worst_source_target_path + [worst_path_length], path_weight_by_targ_by_source)
    
    del path_weight_by_targ_by_source
    
    return worst_source_target_path + [worst_path_length]
    
def draw_graph(G):
    pos = nx.spring_layout(G)
    fig = plt.figure(figsize=(14, 14))
    nx.draw_networkx_nodes(G, pos,node_color='red', node_size=300)
    nx.draw_networkx_labels(G, pos)
    nx.draw_networkx_edges(G, pos)
    plt.show()
    return

falcon_coupling_map = [[0, 1], [1, 0], [1, 2], [1, 4], [2, 1], [2, 3], [3, 2], [3, 5], [4, 1], [4, 7], [5, 3], [5, 8], [6, 7], [7, 4], [7, 6], [7, 10], [8, 5], [8, 9], [8, 11], [9, 8], [10, 7], [10, 12], [11, 8], [11, 14], [12, 10], [12, 13], [12, 15], [13, 12], [13, 14], [14, 11], [14, 13], [14, 16], [15, 12], [15, 18], [16, 14], [16, 19], [17, 18], [18, 15], [18, 17], [18, 21], [19, 16], [19, 20], [19, 22], [20, 19], [21, 18], [21, 23], [22, 19], [22, 25], [23, 21], [23, 24], [24, 23], [24, 25], [25, 22], [25, 24], [25, 26], [26, 25]]

falcon_h_link_pairs = [[26,0]]#[[23,0],[26,3]]
falcon_v_link_pairs = [[9,6],[20,17]]

def graph_from_machine(n_qubits,c_map,add_weights = False, weight_value=1):
    G = nx.Graph()
    G.add_nodes_from(np.arange(n_qubits))
    G.add_edges_from(c_map)
    if add_weights == True:
        nx.set_edge_attributes(G, values = weight_value, name = 'weight')
    return G

def create_square_network_map_27f(edge_length, coupling_map = falcon_coupling_map):
    n_qubits = len(set(deep_flatten(coupling_map)))
    new_coupling_map = []
    
    links = []
    #Modify this line to switch between lower and higher connectivity.
    h_link_pairs = falcon_h_link_pairs
    v_link_pairs = falcon_v_link_pairs
    
    for i in range(edge_length**2):
        for pair in coupling_map:
            temp = []
            temp.append(pair[0]+(i*n_qubits))
            temp.append(pair[1]+(i*n_qubits))
            new_coupling_map.append(temp)
            
            
        if i < edge_length**2-edge_length:
            #add bottom connection via links
            for pair in v_link_pairs:
                temp = []
                temp.append(pair[0]+(i*n_qubits))
                temp.append(pair[1]+(i+edge_length)*n_qubits)
                links.append(temp)
                
                temp = [] #include reverse direction
                temp.append(pair[1]+(i+edge_length)*n_qubits)
                temp.append(pair[0]+(i*n_qubits))
                links.append(temp)
                
        if i%edge_length != (edge_length-1):
            #right connection
            for pair in h_link_pairs:
                temp = []
                temp.append(pair[0]+(i*n_qubits))
                temp.append(pair[1]+((i+1)*n_qubits))
                links.append(temp)
                
                temp = [] #include reverse direction
                temp.append(pair[1]+((i+1)*n_qubits))
                temp.append(pair[0]+(i*n_qubits))
                links.append(temp)
    
    new_coupling_map = new_coupling_map + links
            
    return new_coupling_map, links

octagon_coupling_map = [[0,1], [1,2], [2,3], [3,4], [4,5], [5,6], [6,7], [7,0]]

octagon_joiner_pairs = [[0,5], [1,4], [2, 7], [3,6]]

def create_oct_grid(width_num, height_num, coupling_map = octagon_coupling_map, joiner_pairs=octagon_joiner_pairs):
    """
    Creates a monolithic grid connecting octagons.
    See https://aws.amazon.com/braket/quantum-computers/rigetti/ for source.
    """

    #Finds indices for each octagon & creates internal coupling maps
    octagon_indices = shallow_flatten([[(width_num*j+i) for i in range(width_num)] for j in range(height_num)])
    octagon_internal_couplings = [[[i+index*8, j+index*8]for i,j in coupling_map] for index in octagon_indices]
    #print('\n'.join([str(item) for item in octagon_internal_couplings]))
    """
    
    """
    octagon_cross_couplings = []
    for j in range(height_num):
        for i in range(width_num):
            current_index = width_num*j + i
            current = []
            if j > 0:
                index_up = (j-1)*width_num + i
                current.append([octagon_joiner_pairs[0][0] + current_index*8, octagon_joiner_pairs[0][1]+index_up*8])
                current.append([octagon_joiner_pairs[1][0] + current_index*8, octagon_joiner_pairs[1][1]+index_up*8])
            if i > 0:
                index_left = current_index - 1
                current.append([octagon_joiner_pairs[2][0] + index_left*8 , octagon_joiner_pairs[2][1] + current_index*8])
                current.append([octagon_joiner_pairs[3][0] + index_left*8 , octagon_joiner_pairs[3][1] + current_index*8])
            octagon_cross_couplings.append(current)
    #Done creating links. Are we basically done?
    links = shallow_flatten(octagon_cross_couplings)
    octagon_all_couplings = shallow_flatten(octagon_internal_couplings) + links
    return octagon_all_couplings, links

def create_hex_connects_oct_2(width_num, height_num):
    """
    : rows alternate between falcon & octagon connectivity

    Alternative
        #Might have basic unit as an octagon horizontally linked to two vertically-stacked 27q hex chiplets. This way, the left and right edges each have 2 connections, while the top and bottom each have 4.
        #Link hexagon on left to falcon on right
        hex_octagon_links_internal = [[2, 0+8], [3, 27+8]]
        #Link top falcon with bottom falcon
        falcon_falcon_links_internal = [[pair[0] + 8, pair[1] + 8] for pair in falcon_v_link_pairs]
        #List external links
        links = []
    """
    hexoct_internal_links = [[4,17+8], [5,6+8]]
    hexoct_exposed_links = [[0,9+8], [1,20+8], [2, 7], [3,6], [26+8, 0+8]]
    
    oct_couplings_map_internal = octagon_coupling_map
    hex_couplings_map_internal = [[x+8 for x in pair] for pair in falcon_coupling_map]
    #hexoct_coupling_map_internal = oct_couplings_map_internal + hex_couplings_map_internal# + hexoct_internal_links

    #Now we join this just like joining the octagons, but with different numbers
    hexoct_indices = shallow_flatten([[(width_num*j+i) for i in range(width_num)] for j in range(height_num)])
    all_nonlink_couplings = [[[i+index*35, j+index*35]for i,j in oct_couplings_map_internal + hex_couplings_map_internal]
                             for index in hexoct_indices]
    #print('\n'.join([str(item) for item in octagon_internal_couplings]))
    
    all_cross_couplings = []
    all_internal_links = []
    for j in range(height_num):
        for i in range(width_num):
            current_index = width_num*j + i
            current = []
            if j > 0:
                index_up = (j-1)*width_num + i
                current.append([hexoct_exposed_links[0][0] + current_index*35, hexoct_exposed_links[0][1]+index_up*35])
                current.append([hexoct_exposed_links[1][0] + current_index*35, hexoct_exposed_links[1][1]+index_up*35])
            if i > 0:
                index_left = current_index - 1
                current.append([hexoct_exposed_links[2][0] + index_left*35 , hexoct_exposed_links[2][1] + current_index*35])
                current.append([hexoct_exposed_links[3][0] + index_left*35 , hexoct_exposed_links[3][1] + current_index*35])
                current.append([hexoct_exposed_links[4][0] + index_left*35 , hexoct_exposed_links[4][1] + current_index*35])
            all_cross_couplings.append(current)
            all_internal_links.append([[x+current_index*35 for x in pair]for pair in hexoct_internal_links])
    #Done creating links. Are we basically done?
    links = shallow_flatten(all_internal_links + all_cross_couplings)
    #print(all_internal_links)
    hexoct_all_couplings = shallow_flatten(all_nonlink_couplings) + links
    return hexoct_all_couplings, links

def create_hex_connects_square(width_num, height_num):
    """
    Rows alternate between heavy hex and squares
    """
    hexsq_internal_links = [[2,17+4], [3,6+4]]
    hexsq_exposed_links = [[0,9+4], [1,20+4], [1, 0], [2, 3], [26+4, 0+4]]
    
    sq_couplings_map_internal = [[0,1], [1,2], [2,3], [3,0]]
    hex_couplings_map_internal = [[x+4 for x in pair] for pair in falcon_coupling_map]
    hexsq_coupling_map_internal = sq_couplings_map_internal + hex_couplings_map_internal# + hexoct_internal_links

    #Now we join this just like joining the octagons, but with different numbers
    hexsq_indices = shallow_flatten([[(width_num*j+i) for i in range(width_num)] for j in range(height_num)])
    all_nonlink_couplings = [[[i+index*31, j+index*31]for i,j in sq_couplings_map_internal + hexsq_coupling_map_internal]
                             for index in hexsq_indices]
    #print('\n'.join([str(item) for item in octagon_internal_couplings]))
    
    all_cross_couplings = []
    all_internal_links = []
    for j in range(height_num):
        for i in range(width_num):
            current_index = width_num*j + i
            current = []
            if j > 0:
                index_up = (j-1)*width_num + i
                current.append([hexsq_exposed_links[0][0] + current_index*31, hexsq_exposed_links[0][1]+index_up*31])
                current.append([hexsq_exposed_links[1][0] + current_index*31, hexsq_exposed_links[1][1]+index_up*31])
            if i > 0:
                index_left = current_index - 1
                current.append([hexsq_exposed_links[2][0] + index_left*31 , hexsq_exposed_links[2][1] + current_index*31])
                current.append([hexsq_exposed_links[3][0] + index_left*31 , hexsq_exposed_links[3][1] + current_index*31])
                current.append([hexsq_exposed_links[4][0] + index_left*31 , hexsq_exposed_links[4][1] + current_index*31])
            all_cross_couplings.append(current)
            all_internal_links.append([[x+current_index*31 for x in pair]for pair in hexsq_internal_links])
    #Done creating links. Are we basically done?
    links = shallow_flatten(all_internal_links + all_cross_couplings)
    #print(all_internal_links)
    hexoct_all_couplings = shallow_flatten(all_nonlink_couplings) + links
    return hexoct_all_couplings, links

def create_expander_over_hex():
    """
    Start with (monolithic?) hex, then add some expanderized connections mid-chip?
    """
    

def load_connectivity_from_calibration(calib_file):
    import csv
    
    coupling_set = set()
    
    with open(calib_file) as f:
        reader = csv.DictReader(f)
        for row in reader:
            data = row['CNOT error ']
            pieces = data.split(';')
            for piece in pieces:
                pair = tuple(int(item) for item in piece.split(':')[0].split('_'))
                coupling_set.add(pair)
                
    return coupling_set

    
def create_monolithic_map_heavyhex(n_qubits):
    
    #fix initial values for smallest structure
    q_per_row = 7
    q_connect = 2
    n_rows = 3
    n_connect_rows = n_rows - 1
    dim_found = False
    
    while dim_found == False:
    
        if n_qubits > q_per_row*n_rows + q_connect*n_connect_rows:
            q_per_row = q_per_row + 4
            q_connect = q_connect + 1
            n_rows = n_rows + 2
            n_connect_rows = n_rows - 1
        
        else: 
            dim_found = True
        
    
    
    new_coupling_map = []
    left_align = True
    left_align_vals_back = []
    left_align_vals_fwd = []
    
    for i in range(0,q_connect):
        temp = q_connect + i*3
        left_align_vals_fwd.append(temp)
        left_align_vals_back.append(q_per_row+q_connect-temp)
        
    right_align_vals_back = left_align_vals_fwd.copy()
    right_align_vals_back.reverse()
    
    right_align_vals_fwd = left_align_vals_back.copy()
    right_align_vals_fwd.reverse()
    
    count_q_row = 0
    count_q_connect = 0
    link_back_qubits = []
    for i in range(q_connect):
        link_back_qubits.append([None,None])
    
    for i in range(0,n_qubits):
        
        for item in link_back_qubits:
            if i == item[0]:
                new_coupling_map.append([i,item[1]])
                new_coupling_map.append([item[1],i])    
        
        
        if count_q_row < (q_per_row-1) and i%(q_per_row+q_connect) !=0:
            new_coupling_map.append([i,i-1])
            new_coupling_map.append([i-1,i])
            
            count_q_row = count_q_row + 1 
            
            
        elif count_q_row >= (q_per_row-1):
            if left_align == True:
                new_coupling_map.append([i,i-left_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-left_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [left_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            else:
                #align to right
                new_coupling_map.append([i,i-right_align_vals_back[count_q_connect]])
                new_coupling_map.append([i-right_align_vals_back[count_q_connect],i])
                
                link_back_qubits[count_q_connect] = [right_align_vals_fwd[count_q_connect]+i,i]
                count_q_connect = count_q_connect + 1
                
            if count_q_connect == q_connect:
                count_q_row = 0
                count_q_connect = 0
                left_align = not left_align
    
    G = graph_from_machine(n_qubits,new_coupling_map)
    
    if nx.is_connected(G) == False:
        S = []
        for c in nx.connected_components(G):
            S.append(c.copy())
        
        if len(S[0]) >= len(S[1]):
            new_coupling_map.append([max(S[0]),min(S[1])])
            new_coupling_map.append([min(S[1]),max(S[0])])
            
        else:
            new_coupling_map.append([min(S[0]),max(S[1])])
            new_coupling_map.append([max(S[1]),min(S[0])])    
    
    return new_coupling_map


def create_heavy_hex_chiplet(num_hexes_across, num_row_pairs, switch_start = False):
    """
    Easiest might just be to have rows and joiners
    
    Links should point from right or bottom to left or top
    
    num_down must be a multiple of 2
    
    Vertical couplings look ok, but hor look troubled
    There is no place to split the right edge without having a double-connection!
        This is what motivates the embedded chiplet architecture over hex-preserving.
    """
    row_width = 4*num_hexes_across
    row_joiners_1 = tuple(range(0, num_hexes_across*4, 4))
    row_joiners_2 = tuple(item+2 for item in row_joiners_1)
    row_joiners_all = [row_joiners_1, row_joiners_2]
    
    rows = []
    joiners = []
    row_couplings = []
    row_joiner_couplings = []
    h_links = []
    v_links = []
    offset = 0
    joiner_flag = switch_start
    previous_joiners = None
    
    for row_index in range(num_row_pairs*2):
        row = tuple(range(offset, offset + row_width))
        rows.append(row)
        offset += row_width
        row_couplings.append([(row[i], row[i+1]) for i in range(len(row) - 1)])
        
        h_links.append((row[-1], row[0]))
        
        if previous_joiners is not None:
            #Then we need to attach the row to the previous
            row_joiner_couplings.append([(jq, row[joiner_inds_in_row[i]])
                                         for i, jq in enumerate(previous_joiners)])
        
        joiner_inds_in_row = row_joiners_all[joiner_flag]
        joiner_qubits = tuple(range(offset, offset+len(joiner_inds_in_row)))
        offset += len(joiner_inds_in_row)
        joiners.append(joiner_qubits)
        row_joiner_couplings.append([(row[joiner_inds_in_row[i]], jq)
                                         for i, jq in enumerate(joiner_qubits)])
        previous_joiners = joiner_qubits
        joiner_flag = not joiner_flag
    
    #joiner_inds_in_row = row_joiners_all[joiner_flag]
    for i, jq in enumerate(previous_joiners):
        v_links.append((jq, rows[0][joiner_inds_in_row[i]]))
        
    #Returns row qubits, joiner qubits, (width, height in qubits), copulings, right link locations, bottom link locations
    return (rows, joiners, row_couplings, row_joiner_couplings, h_links, v_links, [], [])
    
def stack_row_joiner_chips(rows_joiners_rcoup_jcoup_hlinks_vlinks_lcr_lcj_1,
                           rows_joiners_rcoup_jcoup_hlinks_vlinks_lcr_lcj_2,
                           vertical):
    """
    Stack across then down, or down then across, make sure respectively widths or heights match
    """
    rows_1, joiners_1, rcoup_1, jcoup_1, hlinks_1, vlinks_1, lcr_1, lcj_1 = rows_joiners_rcoup_jcoup_hlinks_vlinks_lcr_lcj_1
    rows_2, joiners_2, rcoup_2, jcoup_2, hlinks_2, vlinks_2, lcr_2, lcj_2 = rows_joiners_rcoup_jcoup_hlinks_vlinks_lcr_lcj_2
    
    all_qubits_1 = set(deep_flatten(rows_1)) | set(deep_flatten(joiners_1))
    qubit_num_1 = len(all_qubits_1)
    offset = max(all_qubits_1) + 1
    
    def reindex_pairs(col):
        return [(a+offset, b+offset) for a,b in col]
    
    def reindex_pairs_nested(col):
        return [reindex_pairs(sublist) for sublist in col]
    
    def reindex_first_of_pairs(col):
        return [(a+offset, b) for a,b in col]
    
    def reindex_second_of_pairs(col):
        return [(a, b + offset) for a,b in col]
    
    def reindex_qubits(nested_list):
        return [tuple(q + offset for q in sublist) for sublist in nested_list]
    
    
    if vertical:
        #Then we stack under
        rows_new = rows_1 + reindex_qubits(rows_2)
        joiners_new = joiners_1 + reindex_qubits(joiners_2)
        
        rcoup_new = rcoup_1 + reindex_pairs_nested(rcoup_2)
        jcoup_new = jcoup_1 +  reindex_pairs_nested(jcoup_2)
        
        hlinks_new = hlinks_1 + reindex_pairs(hlinks_2)
        vlinks_new = reindex_first_of_pairs(vlinks_2)
        
        lcr_new = lcr_1 + reindex_pairs(lcr_2)
        lcj_new = lcj_1 + reindex_pairs(lcj_2) + reindex_second_of_pairs(vlinks_1)
    else:
        #Then we stack to the right
        rows_new = [row_1 + row_2 for row_1, row_2 in zip(rows_1, reindex_qubits(rows_2))]
        joiners_new = [j1 + j2 for j1, j2 in zip(joiners_1, reindex_qubits(joiners_2))]
        
        rcoup_new = [rc1 + rc2 for rc1, rc2 in zip(rcoup_1, reindex_pairs_nested(rcoup_2))]
        jcoup_new = [jc1 + jc2 for jc1, jc2 in zip(jcoup_1, reindex_pairs_nested(jcoup_2))]
        
        hlinks_new = [(row[-1], row[0]) for row in rows_new]
        vlinks_new = vlinks_1 + reindex_pairs(vlinks_2)
        
        lcr_new = lcr_1 + reindex_pairs(lcr_2) + reindex_second_of_pairs(hlinks_1)
        lcj_new = lcj_1 + reindex_pairs(lcj_2)
        
    return (rows_new, joiners_new, rcoup_new, jcoup_new, hlinks_new, vlinks_new, lcr_new, lcj_new)
    

    
def create_square_network_map_arbitary_q(num_chiplet_hexes_across, num_chiplet_row_pairs,
                                         total_chipnum):
    """
    Going straight down takes 15 jumps, left-to-right takes 11
    
    This saves 3 link uses, so ratio is now 15/60 = 1/4
    
    If we did 120, would get 23 left-to-right or 29 up-down?
        Seems more forward-thinking in a way?
        Up-down would save 9 link total, get ratio of 14/120 = 1/5
    
    We need to (roughly) match total_chipnum in squarest way possible?
    It needs to factor exactly?
        
    We also want to join blocks? Or do that before calling this method?
    """
    chiplet_stats = create_heavy_hex_chiplet(num_chiplet_hexes_across, num_chiplet_row_pairs)
    chip_width = len(chiplet_stats[0][0])
    chip_height = 4*len(chiplet_stats[0])
    
    #1st, optimize for squareness
    best_width_height = (1, total_chipnum)
    for width in range(1, total_chipnum+1):
        height = (total_chipnum // width) + (1 if (total_chipnum % width) else 0)
        #print("Considering %s" % str((width, height)))
        difference = abs(width*chip_width - height*chip_height)
        #print("Difference %d" % difference)
        best_diff = abs(best_width_height[0]*chip_width - best_width_height[1]*chip_height)
        if difference < best_diff:
            best_width_height = (width, height)
    
    #2nd, start stacking
    composite_stats = chiplet_stats
    for i in range(best_width_height[0] - 1):
        composite_stats = stack_row_joiner_chips(composite_stats, chiplet_stats, False)
    full_row_stats = composite_stats
        
    for i in range(best_width_height[1] - 1):
        composite_stats = stack_row_joiner_chips(composite_stats, full_row_stats, True)
    
    return composite_stats


def run_expander_analysis(chip_couplings, chip_joiners, chip_nums,
                          evaluate_pure_expander = True, link_weight = None):
    """
    local weights would be set in chip_couplings, while link_weight could be specified explicitly
    """
    chip_size = len(nodes_from_couplings(chip_couplings))
    degree = len(chip_joiners)
    
    results = []
    for chip_num in chip_nums:
        best_g, best_lambda2 = best_random_expander_chipped(chip_num, 800, 60,
                                                chip_couplings, falcon_joiner_points,
                                                link_weight)
        diam = nx.algorithms.distance_measures.diameter(best_g)
        if evaluate_pure_expander:
            unchipped_n = chip_num*chip_size + ((chip_num*chip_size) % 2)
            best_g_unchipped, best_lambda2_unchipped = best_random_expander(unchipped_n, 3, 100)
            #print("Evaluated chip number %d" % chip_num)
            diam_unchipped = nx.algorithms.distance_measures.diameter(best_g_unchipped)
            results.append([best_g, best_lambda2, diam,
                            best_g_unchipped, best_lambda2_unchipped, diam_unchipped])
        else:
            results.append([best_g, best_lambda2, diam])
    return results

def make_bidir(couplings):
    assert(all(len(pair) == 2 for pair in couplings) or all(len(pair) == 3 for pair in couplings))
    return (set(tuple(pair) for pair in couplings) | set((pair[1], pair[0]) + tuple(pair[2:]) for pair in couplings))

def hex_chip_to_coupling_map(hex_chip, local_link_errors = None):
    rows, joiners, rcoup, jcoup, hlinks, vlinks, lcr, lcj = hex_chip
    
    nodes = set(rows) | set(joiners)
    
    set_merge = lambda s1, s2 : s1 | s2
    
    rcoup_bidir = functools.reduce(set_merge, (make_bidir(s) for s in rcoup))
    jcoup_bidir = functools.reduce(set_merge, (make_bidir(s) for s in jcoup))
    local_couplings = rcoup_bidir | jcoup_bidir
    
    link_couplings = make_bidir(lcr) | make_bidir(lcj)
    
    if local_link_errors is None:
        #Then we run topological analysis
        return (local_couplings | link_couplings)
    
    #Then we should have weighted couplings
    local_error, link_error = local_link_errors
    return [(a,b, local_error) for a,b in local_couplings] + [
        (a,b, link_error) for a,b in link_couplings]

def postproc_27q_weights(couplings, links, local_weight, link_weight):
    new_couplings = set(tuple(item) for item in couplings)
    link_couplings = set(tuple(item) for item in links)
    new_couplings -= link_couplings
    return set((a,b,local_weight) for a,b in new_couplings) + set((a,b,link_weight) for a,b in link_couplings)

def local_grid_couplings(chiplet_side_length, local_weight = None):
    """
    Creates a monolithic grid, which can be itself used as 1 chiplet
    """
    full_couplings = []
    
    for i in range(chiplet_side_length):
        for j in range(chiplet_side_length):
            current_pos = i*chiplet_side_length + j
            if i > 0:
                full_couplings.append([current_pos - chiplet_side_length,current_pos])
            #if i < chiplet_side_length - 1:
            #    full_couplings.append([current_pos, current_pos + chiplet_side_length])
            if j > 0:
                full_couplings.append([current_pos - 1, current_pos])
            #if j < chiplet_side_length - 1:
            #    full_couplings.append([current_pos, current_pos + 1])
    
    if local_weight is not None:
        full_couplings = [[a,b,local_weight] for a,b in full_couplings]
    
    return full_couplings

def grids_q(chiplet_side_length, num_chips_sqrt, local_weight=None, link_weight=None):
    """
    Creates a grid of grids
    """
    chiplet_qubit_num = chiplet_side_length**2
    num_chiplets = num_chips_sqrt**2
    
    KIND_TOP = 0
    KIND_LEF = 1
    KIND_RIG = 2
    KIND_BOT = 3
    #top, left, right, bottom
    link_pts = [chiplet_side_length // 2,
                (chiplet_side_length // 2)*chiplet_side_length,
                (chiplet_side_length // 2)*chiplet_side_length + chiplet_side_length - 1,
                chiplet_qubit_num - ((chiplet_side_length // 2) + 1)]
    
    #Will add weights later
    local = local_grid_couplings(chiplet_side_length, None)
    
    def connection_pt_self_other(i,j,kind):
        self_big_index = i*num_chips_sqrt + j
        self_offset = chiplet_qubit_num*full_index
        
        if kind == KIND_TOP:
            #top, connect to bottom
            other_offset = ((i-1)*num_chips_sqrt + j)*chiplet_qubit_num
            return [link_pts[KIND_TOP] + self_offset, link_pts[KIND_BOT] + other_offset]
        elif kind == KIND_LEF:
            #left, connect to right
            other_offset = (i*num_chips_sqrt + j - 1)*chiplet_qubit_num
            return [link_pts[KIND_LEF] + self_offset, link_pts[KIND_RIG] + other_offset]
        elif kind == KIND_RIG:
            #right
            other_offset = (i*num_chips_sqrt + j + 1)*chiplet_qubit_num
            return [link_pts[KIND_RIG] + self_offset, link_pts[KIND_LEF] + other_offset]
        elif kind == KIND_BOT:
            other_offset = ((i+1)*num_chips_sqrt + j)*chiplet_qubit_num
            return [link_pts[KIND_BOT] + self_offset, link_pts[KIND_TOP] + other_offset]
        
        raise ValueError("No kind %s" % str(kind))
        
    local_couplings = []
    link_couplings = []
    
    for i in range(num_chips_sqrt):
        for j in range(num_chips_sqrt):
            full_index = i*num_chips_sqrt + j
            offset = chiplet_qubit_num*full_index
            #Create local couplings
            local_couplings.append([[a+offset, b+offset] for a,b in local])
            
            #Now add links
            if i > 0:
                link_couplings.append(connection_pt_self_other(i,j,KIND_TOP))
            #if i < num_chips_sqrt - 1:
            #    link_couplings.append(connection_pt_self_other(i,j,KIND_BOT))
            if j > 0:
                link_couplings.append(connection_pt_self_other(i,j,KIND_LEF))
            #if j < num_chips_sqrt - 1:
                #link_couplings.append(connection_pt_self_other(i,j,KIND_RIG))
    #Now to adjust for weight if needed
    if local_weight is not None:
        local_couplings = [[[a,b,local_weight] for a,b in sublist] for sublist in local_couplings]
    if link_weight is not None:
        link_couplings = [[a,b,link_weight] for a,b in link_couplings]
    return [local_couplings, link_couplings]

def tree_of_grids(chiplet_side_length, depth, local_weight=None, link_weight=None):
    local_chip = local_grid_couplings(chiplet_side_length)
    
    connection_to_parent_ind = (0, chiplet_side_length // 2)
    connections_to_children_inds = ((-1, 0),(-1, -1))
    
    def chip_square(local_coup):
        local_chip_nodes = sorted(set(deep_flatten(local_coup)))
        local_chip_square = [local_chip_nodes[i*chiplet_side_length:(i+1)*chiplet_side_length]
                             for i in range(chiplet_side_length)]
        return local_chip_square
    
    def offsetted_local(offset):
        return [[a+offset, b+offset] for a,b in local_chip]
    
    local_couplings = []
    link_couplings = []
    
    g = nx.Graph()
    previous_layer = [local_chip]
    local_couplings.append(previous_layer)
    offset = chiplet_side_length**2
    for i in range(0, depth-1):
        current_layer = []
        for j in range(len(previous_layer)):
            parent_sq = chip_square(previous_layer[j])
            join_pt_left = parent_sq[connections_to_children_inds[0][0]][connections_to_children_inds[0][1]]
            join_pt_righ = parent_sq[connections_to_children_inds[1][0]][connections_to_children_inds[1][1]]
            for pt in [join_pt_left, join_pt_righ]:
                new_local = offsetted_local(offset)
                new_sq = chip_square(new_local)
                link_couplings.append([new_sq[connection_to_parent_ind[0]][connection_to_parent_ind[1]],pt])
                current_layer.append(new_local)
                offset += chiplet_side_length**2
        local_couplings.append(current_layer)
        previous_layer = current_layer
    if local_weight is not None:
        print(local_couplings)
        local_couplings = [[[[a,b,local_weight] for a,b in subsublist] for subsublist in sublist] for sublist in local_couplings]
    if link_weight is not None:
        link_couplings = [[a,b,link_weight] for a,b in link_couplings]
    return (local_couplings, link_couplings)

def plot_graph_value(v_info, y_label):
    """
    v_info should be a list or similar of the form [("name", [(qubit_num, value)])]
    """
    styles = ('-', '--', '-.', ':')
    markers = ['.', 'o', 'v', '^']
    
    plt.figure(figsize=(10,6))
    for i, (name, data) in enumerate(v_info):
        plt.plot([d[0] for d in data], [d[1] for d in data],
                 marker=markers[i % len(markers)], linestyle = styles[i % len(styles)],
                 linewidth=2.5, label=name)
    

    #plt.ylim(0.5,.83)
    plt.xticks(fontsize='xx-large')
    plt.yticks(fontsize='xx-large')
    plt.ylabel(y_label,fontsize='xx-large')
    plt.xlabel('Qubits in System',fontsize='xx-large')
    plt.legend(fontsize='x-large')

falcon_joiner_points = [6,17,9,20,0,26] 

def generate_graphs(names, local_weight = None, link_weight = None):
    chip_nums_squareroots = range(5, 11) #range(3,11)
    octagon_nums_squareroots = range(10, 16)
    hexoct_nums_squareroots = range(3,8)
    hexsq_nums_squareroots = range(3, 9)
    tree_depths = range(4, 8) #range(2, 8)
    chiplet_large_chip_nums = [9, 12, 16, 20, 25, 30, 36]
    chip_nums = [item**2 for item in chip_nums_squareroots]
    qubit_nums = [chip_num*27 for chip_num in chip_nums]
    
    graphs_to_consider = []
    
    for name in names:
        if name == "Mono Hex":
            if local_weight is None: 
                graphs = [graph_from_couplings(
                create_monolithic_map_heavyhex(qubit_num)) for qubit_num in qubit_nums]
            else:
                graphs = [graph_from_couplings([(a,b,local_weight) for a,b in
                    create_monolithic_map_heavyhex(qubit_num)]) for qubit_num in qubit_nums]
        elif name == "Mono Grid":
            mono_grids = [local_grid_couplings(chip_num_sq*5, local_weight) for chip_num_sq in chip_nums_squareroots]
            graphs = [graph_from_couplings(make_bidir(grid)) for grid in mono_grids]
        elif name == "Chiplet Grid":
            grid_chiplet_couplings = [make_bidir(shallow_flatten(local_coups) + link_coups)
                                      for local_coups, link_coups in [grids_q(
                                          5, chip_num_sq, local_weight, link_weight)
                                                                      for chip_num_sq in chip_nums_squareroots]]
            graphs = [graph_from_couplings(coup) for coup in grid_chiplet_couplings]
        elif name == "Tree of Grids":
            grid_trees = [tree_of_grids(5, depth, local_weight, link_weight) for depth in tree_depths]
            grid_tree_coups = [make_bidir(shallow_flatten(shallow_flatten(loc))) | make_bidir(links)
                               for loc, links in grid_trees]
            graphs = [graph_from_couplings(coup) for coup in grid_tree_coups]
        elif name == "27q Chiplet":
            chiplet_27q = [create_square_network_map_27f(chip_num_sqrt) for chip_num_sqrt in chip_nums_squareroots]
            if local_weight is None:
                graphs = [graph_from_couplings(couplings) for couplings, links in chiplet_27q]
            else:
                graphs = [graph_from_couplings(postproc_27q_weights(couplings, links))
                              for couplings, links in weighted_27q]
        elif name == "Octagons":
            oct_couplings_links = [create_oct_grid(num, num) for num in octagon_nums_squareroots]
            if local_weight is None:
                graphs = [graph_from_couplings(couplings) for couplings, links in oct_couplings_links]
            else:
                graphs = [graph_from_couplings(postproc_27q_weights(couplings, links))
                              for couplings, links in oct_couplings_links]
        elif name == "Hex-Octagon":
            hexoct_couplings_links = [create_hex_connects_oct_2(num, num) for num in hexoct_nums_squareroots]
            #print(hexoct_couplings_links[0])
            if local_weight is None:
                graphs = [graph_from_couplings(couplings) for couplings, links in hexoct_couplings_links]
            else:
                graphs = [graph_from_couplings(postproc_27q_weights(couplings, links))
                              for couplings, links in hexoct_couplings_links]
        elif name == "Hex-Square":
            hexsq_couplings_links = [create_hex_connects_square(num, num) for num in hexsq_nums_squareroots]
            #print(hexoct_couplings_links[0])
            if local_weight is None:
                graphs = [graph_from_couplings(couplings) for couplings, links in hexsq_couplings_links]
            else:
                graphs = [graph_from_couplings(postproc_27q_weights(couplings, links))
                              for couplings, links in hexsq_couplings_links]
        elif name == "80q Hex Chiplet":
            chiplet_large_hex_across_num = 4
            chiplet_large_row_pair_num = 2
            chiplet_large_qubit_base_num = 80
            structures_chiplet_large_q = [create_square_network_map_arbitary_q(
                chiplet_large_hex_across_num, chiplet_large_row_pair_num,  chip_num)
                for chip_num in chiplet_large_chip_nums]
            if local_weight is None:
                graphs = [graph_from_couplings(hex_chip_to_coupling_map(hex_chip))
                                        for hex_chip in structures_chiplet_large_q]
            else:
                graphs = [graph_from_couplings(hex_chip_to_coupling_map(hex_chip, (local_weight, link_weight)))
                               for hex_chip in structures_chiplet_large_q]
        elif name == "Expander":
            if local_weight is None:
                exp_res = run_expander_analysis(falcon_coupling_map, falcon_joiner_points, chip_nums,
                                                evaluate_pure_expander = False)
            else:
                weighted_falcon_couplings = [[a,b,local_weight] for a,b in falcon_coupling_map]
                exp_res = run_expander_analysis(weighted_falcon_couplings, falcon_joiner_points,
                                                chip_nums, evaluate_pure_expander = False,
                                                link_weight = link_weight)
            graphs = [item[0] for item in exp_res]
            #graphs_to_consider.append(("Expanded Chiplet", graphs_chipped_expander))
            #graphs_pure_expander = [item[3] for item in exp_res]
            #graphs_to_consider.append(("Full Expander", graphs_pure_expander))
        graphs_to_consider.append((name, graphs))
    #Done with main loop
    return graphs_to_consider

def process_graphs(graphs_to_consider):
    """
    Want to test 4 cfgs: monolithic, connected Falcon, 
    """
    
    #Now to obtain the desired metrics on all this mess
    metrics_results = []
    for name, graphs in graphs_to_consider:
        print(name)
        current_list = []
        metrics_results.append((name, current_list))
        for g in graphs:
            qubit_num = len(g)
            source, targ, path, diam = find_worst_path(g)
            assert(nx.algorithms.components.is_connected(g))
            lambda2 = graph_lambda2(g)
            print("Qubit Number %d, Diameter %f, Lambda2 %f " % (qubit_num, diam, lambda2))
            current_list.append((qubit_num, [source, targ, path, diam], lambda2))
        print()
    return metrics_results





