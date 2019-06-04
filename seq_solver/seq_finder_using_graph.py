import numpy as np
from pattern_discovery.display.raster import plot_spikes_raster
import networkx as nx
import networkx.algorithms.dag as dag
import time
from pattern_discovery.graph.force_directed_graphs import plot_graph_using_fa2
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra


def build_mle_transition_dict(spike_nums, min_duration_intra_seq, time_inter_seq, debug_mode=False,
                              with_dist=True):
    """
    Maximum Likelihood estimation,
    don't take into account the fact that if a neuron A fire after a neuron B ,
    then it decreases the probability than B fires after A
    :param spike_nums:
    :param param:
    :param with_dist: if True, return a matrix representing the dist between spikes
    :return:
    """
    if debug_mode:
        print("building Maximum Likelihood estimation transition dict")
    start_time = time.time()
    nb_neurons = len(spike_nums)
    n_times = len(spike_nums[0, :])
    transition_dict = np.zeros((nb_neurons, nb_neurons))
    # give the average distance between consecutive spikes of 2 neurons, in frames
    if with_dist:
        spikes_dist_dict = np.zeros((nb_neurons, nb_neurons))
        # count to make average
        spikes_count_dict = np.zeros((nb_neurons, nb_neurons))
    # so the neuron with the lower spike rates gets the biggest weight in terms of probability
    spike_rates = np.ones(nb_neurons)

    # a first round to put probabilities up from neurons B that spikes after neuron A
    for neuron_index, neuron_spikes in enumerate(spike_nums):

        # will count how many spikes of each neuron are following the spike of
        for t in np.where(neuron_spikes)[0]:
            # print(f"min_duration_intra_seq {min_duration_intra_seq}")
            t_min = np.max((0, t + min_duration_intra_seq))
            t_max = np.min((t + time_inter_seq, n_times))
            times_to_check = np.arange(t_min, t_max)

            actual_neurons_spikes = spike_nums[neuron_index, :] > 0
            # removing the spikes so they are not found in the later search
            spike_nums[neuron_index, actual_neurons_spikes] = 0

            # Retrieve all cells active during the period of time times_to_check
            if len(times_to_check) == 1:
                co_active_cells = np.where(spike_nums[:, times_to_check])[0]
            else:
                # co-active cells
                co_active_cells = np.where(np.sum(spike_nums[:, times_to_check], axis=1))[0]

            # pos = np.unique(pos)
            for p in co_active_cells:
                transition_dict[neuron_index, p] = transition_dict[neuron_index, p] + \
                                                   spike_rates[p]

                first_spike_pos = np.where(spike_nums[p, times_to_check])[0][0]
                first_spike_pos += min_duration_intra_seq
                if with_dist:
                    spikes_dist_dict[neuron_index, p] += first_spike_pos
                    spikes_count_dict[neuron_index, p] += 1

            # back to one
            spike_nums[neuron_index, actual_neurons_spikes] = 1

        transition_dict[neuron_index, neuron_index] = 0

    # try normalizing by the mean spike count of the 2 cells
    normalize_by_spike_count = False
    if normalize_by_spike_count:
        for cell_1 in np.arange(nb_neurons):
            for cell_2 in np.arange(nb_neurons):
                if cell_1 == cell_2:
                    continue
                cell_1_count = len(np.where(spike_nums[cell_1])[0])
                # cell_2_count = len(np.where(spike_nums[cell_2])[0])
                # mean_spikes_count = (cell_1_count + cell_2_count) / 2
                if cell_1_count > 0:
                    transition_dict[cell_1, cell_2] = transition_dict[cell_1, cell_2] / \
                                                      cell_1_count

    # all negatives values should be put to zero
    transition_dict[np.where(transition_dict < 0)] = 0
    # more elegant way, but the issue is when the sum equal 0
    # transition_dict = transition_dict / transition_dict.sum(axis=1, keepdims=True)

    keeping_the_nb_of_rep = True
    if not keeping_the_nb_of_rep:
        # we divide for each neuron the sum of the probabilities to get the sum to 1
        for neuron_index in np.arange(nb_neurons):
            if np.sum(transition_dict[neuron_index, :]) > 0:
                transition_dict[neuron_index, :] = transition_dict[neuron_index, :] / \
                                                   np.sum(transition_dict[neuron_index, :])
            else:
                print(f"For cell {neuron_index}, transition_dict is 0, n_spikes: {np.sum(spike_nums[neuron_index])}")

    print_transit_dict = False
    if print_transit_dict:
        for neuron_index in np.arange(nb_neurons):
            print(f'transition dict, n {neuron_index}, sum: {np.sum(transition_dict[neuron_index, :])}')
            print(f'transition dict, n {neuron_index}, max: {np.max(transition_dict[neuron_index, :])}')
            print(f'transition dict, n {neuron_index}, nb max: '
                  f'{np.where(transition_dict[neuron_index, :] == np.max(transition_dict[neuron_index, :]))[0]}')
    # if debug_mode:
    #     print(f'median transition: {np.median(transition_dict)}')
    #     print(f'mean transition: {np.mean(transition_dict)}')
    #     print(f'std transition: {np.std(transition_dict)}')
    #     print(f'min transition: {np.min(transition_dict)}')
    #     print(f'max transition: {np.max(transition_dict)}')
    if debug_mode:
        stop_time = time.time()
        print(f"Maximum Likelihood estimation transition dict built in {np.round(stop_time - start_time, 3)} s")

    if with_dist:
        # averaging
        spikes_count_dict[spikes_count_dict == 0] = 1
        spikes_dist_dict = np.divide(spikes_dist_dict, spikes_count_dict)

        return transition_dict, spikes_dist_dict
    else:
        return transition_dict


def get_seq_times_from_raster(raster, min_time_bw_2_spikes, max_time_bw_2_spikes, error_rate,
                              max_errors_in_a_row=3, min_len_ratio=None,
                              min_seq_len=None, cell_indices=None):
    """
    Raster represents the len of seq (shape[0]), the fct will return a list of tuple of size (shape[0]),
    with an int > 0 at the frames where a cell fire in a seq, and -1 if the cell don't fire in the seq
    :param raster:
    :param min_time_bw_2_spikes
    :param max_time_bw_2_spikes
    :param error_rate:
    :param min_len_ratio: min number of cells with spikes in the seq, ratio comparing to the total number of cells
    used only if min_seq_len is None
    :param min_seq_len: min number of cells with spikes in the seq
    :return:
    """
    # TODO: keep for each set of times, the cells associated
    # because co-active cells could have been added to sequences, we extend the search
    min_time_bw_2_spikes = - 10
    max_time_bw_2_spikes = 25
    raster = np.copy(raster)
    n_cells = raster.shape[0]
    n_times = raster.shape[1]
    max_n_errors = int(n_cells * error_rate)
    min_time_bw_2_spikes_original = min_time_bw_2_spikes
    max_time_bw_2_spikes_original = max_time_bw_2_spikes
    all_seq_times = []
    # dict to keep the number of rep of each seq depending on the cells it is composed from
    seq_times_by_seq_cells_dict = dict()
    if min_seq_len is None:
        if min_len_ratio is None:
            min_len_ratio = 0.4
        min_seq_len = max(2, int(n_cells * min_len_ratio))
        if n_cells >= 5:
            min_seq_len = max(3, min_seq_len)

    for cell in np.arange(n_cells):
        if cell > (n_cells-min_seq_len):
            # means the number of errors is already too high to find any significant sequences
            break
        cell_spike_times = np.where(raster[cell])[0]
        # last_raster_copy = np.copy(raster)
        for cell_spike_time in cell_spike_times:
            last_raster_copy = np.copy(raster)
            min_time_bw_2_spikes = min_time_bw_2_spikes_original
            max_time_bw_2_spikes = max_time_bw_2_spikes_original
            # n_errors = cell
            # first cells don't matter, what matters are the errors in the middle
            n_errors = 0
            n_errors_in_a_raw = 0
            times_in_seq = [-1]*cell + [cell_spike_time]
            # removing spikes one after the other in order not to find them again
            raster[cell, cell_spike_time] = 0
            last_spike_time = cell_spike_time
            for following_cell in np.arange(cell + 1, n_cells):
                t_min = np.max((0, last_spike_time + min_time_bw_2_spikes))
                t_min = np.min((n_times, t_min))
                t_max = np.min((n_times, last_spike_time + max_time_bw_2_spikes))
                if t_min == t_max:
                    break
                # print(f"following_cell {following_cell}, t_min {t_min}, t_max {t_max}")
                following_cells_spikes = np.where(raster[following_cell, t_min:t_max])[0]
                if len(following_cells_spikes) == 0:
                    # if we didn't reach the max number of errors, we go to the next cell
                    if (n_errors < max_n_errors) and (n_errors_in_a_raw < max_errors_in_a_row):
                        n_errors += 1
                        max_errors_in_a_row += 1
                        # the next cell spikes has a wider choice for spiking, has we don't have the one in the middle
                        # TODO: if other seq have been detected before, we could use the spike intervals in those
                        # TODO: to estimate where to look
                        min_time_bw_2_spikes = int(min_time_bw_2_spikes * 1.2)
                        max_time_bw_2_spikes = int(max_time_bw_2_spikes * 1.2)
                        times_in_seq.append(-1)
                        continue
                    else:
                        times_in_seq.extend([-1]*(n_cells-following_cell))
                        break
                else:
                    max_errors_in_a_row = 0
                    following_cell_spike_time = following_cells_spikes[0]
                    following_cell_spike_time += last_spike_time + min_time_bw_2_spikes
                    following_cell_spike_time = np.min((n_times, following_cell_spike_time))
                    following_cell_spike_time = np.max((0, following_cell_spike_time))
                    last_spike_time = following_cell_spike_time
                    # back to normal time intervals
                    min_time_bw_2_spikes = min_time_bw_2_spikes_original
                    max_time_bw_2_spikes = max_time_bw_2_spikes_original
                    # removing spikes one after the other in order not to find them again
                    raster[following_cell, following_cell_spike_time] = 0
                    times_in_seq.append(following_cell_spike_time)
            # checking if we have a full seq and at least half of the cells have a real spike
            n_spikes = len(np.where(np.array(times_in_seq) >= 0)[0])
            if n_spikes >= min_seq_len:
                times_in_seq += [-1] * (n_cells - len(times_in_seq))
                # print(f"times_in_seq len: {len(times_in_seq)}")
                all_seq_times.append(times_in_seq)
                times_in_seq = np.array(times_in_seq)
                if cell_indices is not None:
                    cells_indices_with_spikes = np.where(times_in_seq >= 0)[0]
                    cells_associated = tuple(cell_indices[cells_indices_with_spikes])
                    if cells_associated not in seq_times_by_seq_cells_dict:
                        seq_times_by_seq_cells_dict[cells_associated] = []
                    seq_times_by_seq_cells_dict[cells_associated].append(times_in_seq[times_in_seq >= 0])
            else:
                # we put back the spikes erased that didn't give a sequence
                raster = last_raster_copy
    # print(f"min_seq_len {min_seq_len}, n_cells {n_cells}")
    return all_seq_times, seq_times_by_seq_cells_dict


def get_weight_of_a_graph_path(graph, path):
    total_weight = 0
    for cell_index, cell in enumerate(path):
        if cell_index == len(path) - 1:
            break
        weight = graph[cell][path[cell_index + 1]]['weight']
        total_weight += weight
    return total_weight


def build_graph_from_transition_dict(transition_dict, n_connected_cell_to_add,
                                     use_longest_path=False, with_weight=True,
                                     cells_to_isolate=None, transition_dict_2_nd_order=None,
                                     spikes_dist_dict=None, min_rep_nb=None):
    """

    :param transition_dict: a 2d np.array, should be square
    :param n_connected_cell_to_add: n-th best conencted cell to be linked to
    :param use_longest_path: not using shortest path, but longest path, then the graph will be acyclic
    :param with_weight: using weight, the weight would be the rank in the transition dict (sorted)
    :param cells_to_isolate: list of cells that should be include in the graph, and should be returned as isolated
    :return:
    """
    n_cells = transition_dict.shape[0]
    graph = nx.DiGraph()
    graph.add_nodes_from(np.arange(n_cells))
    list_cells_connected = set()
    if cells_to_isolate is None:
        cells_to_isolate = np.zeros(0)
    for cell in np.arange(n_cells):
        if np.sum(transition_dict[cell, :]) == 0:
            # it usually means the cells has no transients
            continue
        # cell to isolate, but can follow an other cell in the directed graph
        if cell in cells_to_isolate:
            continue

        # sorting the cells from the most connected to the less
        connected_cells_sorted = np.argsort(transition_dict[cell, :])[::-1]
        values_sorted = np.sort(transition_dict[cell, :])[::-1]
        # if some values are at zero, we don't include them
        connected_cells_sorted = connected_cells_sorted[values_sorted > 0]
        if len(connected_cells_sorted) == 0:
            continue
        # removing cells_to_isolate
        if len(cells_to_isolate) > 0:
            connected_cells_sorted_tmp = connected_cells_sorted
            connected_cells_sorted = []
            for cell_connec in connected_cells_sorted_tmp:
                if cell_connec not in cells_to_isolate:
                    connected_cells_sorted.append(cell_connec)
            connected_cells_sorted = np.array(connected_cells_sorted)
        if len(connected_cells_sorted) == 0:
            continue
        if min_rep_nb is not None:
            # for this to work, first_cell_transition_score should correspond to the nb of rep
            # if none of the link is repeated enough then we don't take it into consideration
            first_cell_transition_score = transition_dict[cell, connected_cells_sorted[0]]
            if first_cell_transition_score < min_rep_nb:
                continue

        # if more than one cell are at the same distance than the cell we're looking at
        # we link those cells sorted by the distance of their spikes from the main cell
        link_co_active_cells = True
        if link_co_active_cells:
            first_cell_transition_score = transition_dict[cell, connected_cells_sorted[0]]
            cells_with_same_score = np.where(transition_dict[cell, connected_cells_sorted[1:]] ==
                                             first_cell_transition_score)[0]
            if len(cells_with_same_score) > 0:
                cells_with_same_score = connected_cells_sorted[:len(cells_with_same_score) + 1]
                # we sort them by dist:
                if spikes_dist_dict is not None:
                    sorted_by_dist = np.argsort(spikes_dist_dict[cell, cells_with_same_score])
                    cells_with_same_score = cells_with_same_score[sorted_by_dist]
                last_cell = cell
                for new_cell in cells_with_same_score:
                    # we want to connect those cells one by one
                    if with_weight:
                        graph.add_edge(last_cell, new_cell, weight=1)
                    else:
                        graph.add_edge(last_cell, new_cell)
                    last_cell = new_cell
                cells_to_isolate = np.concatenate((cells_to_isolate, cells_with_same_score[:-1]))
                continue
        # else:
        #     connected_cells_sorted = connected_cells_sorted[:n_connected_cell_to_add]
        # cell_to_check=25
        # if cell == cell_to_check:
        #     print(f"{cell_to_check}: connected_cells_sorted {' '.join(map(str, connected_cells_sorted))}")
        #     print(f"{cell_to_check}: transition_dict {' '.join(map(str, transition_dict[cell, connected_cells_sorted]))}")
        #     print(f"{cell_to_check}: transition_dict std {np.std(transition_dict[cell, :])}")
        #     print(f"{cell_to_check}: spikes_dist_dict {' '.join(map(str, spikes_dist_dict[cell, connected_cells_sorted]))}")

        # this part is useful if link_co_active_cells is False. Will order cells with the same score
        # accordingly to their distance and the 2nd order transition_dict
        connected_cells_sorted = connected_cells_sorted[:n_connected_cell_to_add]
        if n_connected_cell_to_add > 1:
            first_cell_transition_score = transition_dict[cell, connected_cells_sorted[0]]
            cells_with_same_score = np.where(transition_dict[cell, connected_cells_sorted[1:]] ==
                                             first_cell_transition_score)[0]
            if len(cells_with_same_score) > 0:
                # we order them by distance
                if spikes_dist_dict is not None:
                    cells_to_sort = connected_cells_sorted[:len(cells_with_same_score) + 1]
                    sorted_by_dist = np.argsort(spikes_dist_dict[cell, cells_to_sort])
                    cells_to_sort = cells_to_sort[sorted_by_dist]
                    connected_cells_sorted[:len(cells_with_same_score) + 1] = cells_to_sort
                elif transition_dict_2_nd_order is not None:
                    # we could start by looking to which cells is connected cell, and using a '2nd'
                    # order transition_dict,
                    # order the connected cell according to their rank in the 2nd order dict
                    cells_to_sort = connected_cells_sorted[:len(cells_with_same_score) + 1]
                    sorted_by_dist = np.argsort(transition_dict_2_nd_order[cell, cells_to_sort])[::-1]
                    cells_to_sort = cells_to_sort[sorted_by_dist]
                    connected_cells_sorted[:len(cells_with_same_score) + 1] = cells_to_sort
            else:
                if transition_dict_2_nd_order is not None:
                    # we could start by looking to which cells is connected cell, and using a '2nd'
                    # order transition_dict,
                    # order the connected cell according to their rank in the 2nd order dict
                    sorted_2nd_order = np.argsort(transition_dict_2_nd_order[cell, connected_cells_sorted])[::-1]
                    connected_cells_sorted = connected_cells_sorted[sorted_2nd_order]

        # if cell == cell_to_check:
        #     print(f"2nd order {cell_to_check}: connected_cells_sorted {' '.join(map(str, connected_cells_sorted))}")
        for connex_index, cell_connected in enumerate(connected_cells_sorted):
            if (cell == cell_connected):
                continue
            # threshold_prob = (mean_trans_dict + (n_std_for_threshold * std_trans_dict))
            # threshold_prob = np.median(transition_dict[cell, :])
            # we add it, only if it passes a probability threshold
            # if transition_dict[cell, cell_connected] <= threshold_prob:
            #     # print(f"Stop edges process {cell} -> {cell_connected} "
            #     #       f"at step {connex_index}: {str(np.round(threshold_prob, 4))}")
            #     break
            if use_longest_path:
                # to make sure the graph is acyclic
                if cell_connected in list_cells_connected:
                    continue
                list_cells_connected.add(cell_connected)
                list_cells_connected.add(cell)
            if with_weight:
                # the cell the most connected has the bigger weight
                # for the weight we could use: 1 - transition_dict[cell, cell_connected]
                # we use the rank from this cell, the aim is to use it for shortest path
                # the shortest weighted path, will be the one with the higher probability
                graph.add_edge(cell, cell_connected, weight=connex_index + 1)
                # graph.add_edge(cell, cell_connected, weight=1 - transition_dict[cell, cell_connected])
            else:
                # print(f"add_edge {(cell, cell_connected)}")
                graph.add_edge(cell, cell_connected)

    return graph


def find_paths_in_a_graph(graph, shortest_path_on_weight, with_weight, use_longest_path=False,
                          debug_mode=False):
    # first we remove cell with no neighbors
    isolates_cell = []
    seq_list = []
    while True:
        new_isolate_cells = list(nx.isolates(graph))
        graph.remove_nodes_from(new_isolate_cells)
        isolates_cell.extend(new_isolate_cells)
        if graph.number_of_nodes() == 0:
            break
        if use_longest_path:
            longest_shortest_path = dag.dag_longest_path(graph)
            if debug_mode:
                print(f"longest_path {len(longest_shortest_path)}: {longest_shortest_path}")
        else:
            longest_shortest_path = []
            lowest_weight_among_best_path = None
            if shortest_path_on_weight:
                for cell, (dist_dict, path_dict) in nx.all_pairs_dijkstra(graph):
                    for target_cell, path in path_dict.items():
                        if len(path) >= len(longest_shortest_path):
                            weight = get_weight_of_a_graph_path(graph, list(path))
                            if len(path) > len(longest_shortest_path):
                                longest_shortest_path = list(path)
                                lowest_weight_among_best_path = weight
                                if debug_mode:
                                    print(f"{list(path)}: {weight}")
                            # else both the same length, then we look at the weight
                            elif lowest_weight_among_best_path > weight:
                                lowest_weight_among_best_path = weight
                                if debug_mode:
                                    print(f"{list(path)}: {weight}")
            else:
                shortest_paths_dict = dict(all_pairs_shortest_path(graph))
                # for weighted choice: all_pairs_dijkstra(G)
                for node_1, node_2_dict in shortest_paths_dict.items():
                    for node_2, path in node_2_dict.items():
                        if len(path) >= len(longest_shortest_path):
                            if with_weight:
                                weight = get_weight_of_a_graph_path(graph, list(path))
                            else:
                                weight = 0
                            if len(path) > len(longest_shortest_path):
                                longest_shortest_path = list(path)
                                lowest_weight_among_best_path = weight
                                if debug_mode:
                                    print(f"{list(path)}: {weight}")
                            # else both the same length, then we look at the weight
                            elif lowest_weight_among_best_path > weight:
                                lowest_weight_among_best_path = weight
                                if debug_mode:
                                    print(f"{list(path)}: {weight}")
                if debug_mode:
                    print(f"longest_shortest_path {len(longest_shortest_path)}: {longest_shortest_path} "
                      f"{lowest_weight_among_best_path}")
        seq_list.append(longest_shortest_path)
        graph.remove_nodes_from(longest_shortest_path)

        if graph.number_of_nodes() == 0:
            break

    return seq_list, isolates_cell


def find_sequences_using_graph_main(spike_nums, param, min_time_bw_2_spikes, max_time_bw_2_spikes,
                               n_surrogates,
                               max_connex_by_cell, min_nb_of_rep=None,
                                    debug_mode=False, descr="", ms=None):
    # spike_nums_backup = spike_nums
    spike_nums = np.copy(spike_nums)
    # spike_nums = spike_nums[:, :2000]
    n_cells = spike_nums.shape[0]
    n_times = spike_nums.shape[1]

    transition_dict, spikes_dist_dict = build_mle_transition_dict(spike_nums=spike_nums,
                                                                  min_duration_intra_seq=min_time_bw_2_spikes,
                                                                  time_inter_seq=max_time_bw_2_spikes,
                                                                  debug_mode=True)

    if n_surrogates > 0:
        start_time = time.time()
        print(f"starting to generate {n_surrogates} surrogates")
        all_surrogate_transition_dict = np.zeros((n_cells, n_cells, n_surrogates))
        for surrogate_index in np.arange(n_surrogates):
            surrogate_spike_nums = np.copy(spike_nums)
            for cell, neuron_spikes in enumerate(surrogate_spike_nums):
                # roll the data to a random displace number
                surrogate_spike_nums[cell, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            t_d = build_mle_transition_dict(spike_nums=surrogate_spike_nums,
                                            min_duration_intra_seq=min_time_bw_2_spikes,
                                            time_inter_seq=max_time_bw_2_spikes,
                                            debug_mode=False, with_dist=False)
            all_surrogate_transition_dict[:, :, surrogate_index] = t_d
        # surrogate_threshold_transition_dict = np.zeros((n_cells, n_cells))
        n_values_removed = 0
        for cell_1 in np.arange(n_cells):
            for cell_2 in np.arange(n_cells):
                # surrogate_threshold_transition_dict[cell_1, cell_2] = \
                #     np.percentile(all_surrogate_transition_dict[cell_1, cell_2, :], 95)
                if cell_1 == cell_2:
                    continue
                surrogate_value = np.percentile(all_surrogate_transition_dict[cell_1, cell_2, :], 95)
                if surrogate_value >= transition_dict[cell_1, cell_2]:
                    n_values_removed += 1
                    transition_dict[cell_1, cell_2] = 0
        print(f"{n_values_removed} values removed from transition_dict after surrogates "
              f"{np.round((n_values_removed / (cell_1*cell_2))*100, 2)} %")
        stop_time = time.time()
        print(f"Time to generate surrogates {np.round(stop_time - start_time, 3)} s")

    transition_dict_2_nd_order = None
    try_with_2nd_order = True
    if try_with_2nd_order:
        print("Building 2nd order transition dict")
        transition_dict_2_nd_order, \
        spikes_dist_dict_2_nd_order = \
            build_mle_transition_dict(spike_nums=spike_nums,
                                      min_duration_intra_seq=min_time_bw_2_spikes * 2,
                                      time_inter_seq=max_time_bw_2_spikes * 2,
                                      debug_mode=debug_mode)

    try_graph_solution = True
    if try_graph_solution:
        # qualitative 12 colors : http://colorbrewer2.org/?type=qualitative&scheme=Paired&n=12
        # + 11 diverting
        colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c', '#fdbf6f',
                  '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928', '#a50026', '#d73027',
                  '#f46d43', '#fdae61', '#fee090', '#ffffbf', '#e0f3f8', '#abd9e9',
                  '#74add1', '#4575b4', '#313695']
        plot_graph = False
        with_weight = False
        # if true, means that the algorithm will return between all pairs, the path that has the lowest weight
        # and not necessarly the one with the less cells on the path
        shortest_path_on_weight = False
        use_longest_path = False
        use_graph_from_connectity_ms = False
        n_connected_cell_to_add = max_connex_by_cell
        if use_graph_from_connectity_ms and (ms is not None):
            print('use_graph_from_connectity_ms')
            ms.detect_n_in_n_out()
            graph = ms.spike_struct.graph_out
        else:
            # cells_to_isolate = None
            # removing from the graph cells that fires the most and the less
            spike_count_by_cell = np.sum(spike_nums, axis=1)
            low_spike_count_threshold = np.percentile(spike_count_by_cell, 10)
            cells_to_isolate = np.where(spike_count_by_cell < low_spike_count_threshold)[0]
            high_spike_count_threshold = np.percentile(spike_count_by_cell, 95)
            cells_to_isolate = np.concatenate((cells_to_isolate,
                                               np.where(spike_count_by_cell > high_spike_count_threshold)[0]))
            # the idea is to build a graph based on top n connection from the transition_dict
            # directed graph
            graph = build_graph_from_transition_dict(transition_dict, n_connected_cell_to_add,
                                                     use_longest_path=use_longest_path,
                                                     with_weight=with_weight, cells_to_isolate=cells_to_isolate,
                                                     transition_dict_2_nd_order=transition_dict_2_nd_order,
                                                     spikes_dist_dict=spikes_dist_dict,
                                                     min_rep_nb=min_nb_of_rep)

        if plot_graph:
            plot_graph_using_fa2(graph=graph, file_name=f"graph_seq",
                                 title=f"graph_seq",
                                 param=param, iterations=15000, save_raster=True, with_labels=False,
                                 save_formats="pdf", show_plot=False)
        # cycles = nx.simple_cycles(graph)
        # print(f"len(cycles) {len(list(cycles))}")
        if debug_mode:
            print(f"dag.is_directed_acyclic_graph(graph) {dag.is_directed_acyclic_graph(graph)}")

        seq_list, isolates_cell = find_paths_in_a_graph(graph, shortest_path_on_weight, with_weight,
                                                        use_longest_path=False, debug_mode=debug_mode)

        # we have a list of seq that we want to concatenate according to the score of transition between
        # the first and last cell in transition dict
        # first we keep aside the seq that are composed of less than 3 cells
        short_seq_cells = []
        long_seq_list = []
        for seq in seq_list:
            # if len(seq) <= 2:
            #     # organize those ones according to transition dict
            #     if len(seq) == 1:
            #         print(f"len(seq) == 1: {seq}")
            #     short_seq_cells.extend(seq)
            # else:
            #     print(f"long_seq_list.append {seq}")
            long_seq_list.append(seq)
        n_long_seq = len(long_seq_list)
        seq_transition_dict = np.zeros((n_long_seq, n_long_seq))
        for long_seq_index_1, long_seq_1 in enumerate(long_seq_list):
            for long_seq_index_2, long_seq_2 in enumerate(long_seq_list):
                if long_seq_index_1 == long_seq_index_2:
                    continue
                first_cells = long_seq_1[-2:]
                following_cells = long_seq_2[:2]
                # best_tuple = None
                # best_transition_prob = 0
                sum_prob = 0
                sum_prob += transition_dict[first_cells[1], following_cells[0]]
                if transition_dict_2_nd_order is not None and (spikes_dist_dict_2_nd_order is not None):
                    dist_1_st_order = spikes_dist_dict[first_cells[0], first_cells[1]]
                    dist_2_nd_order = spikes_dist_dict_2_nd_order[first_cells[0], following_cells[0]]
                    if dist_2_nd_order > dist_1_st_order + param.min_duration_intra_seq:
                        sum_prob = max(sum_prob, transition_dict_2_nd_order[first_cells[0], following_cells[0]])

                    dist_1_st_order = spikes_dist_dict[first_cells[1], following_cells[0]]
                    dist_2_nd_order = spikes_dist_dict_2_nd_order[first_cells[1], following_cells[1]]
                    if dist_2_nd_order > dist_1_st_order + param.min_duration_intra_seq:
                        sum_prob = max(sum_prob, transition_dict_2_nd_order[first_cells[1], following_cells[1]])

                # for first_cell in first_cells:
                #     for following_cell in following_cells:
                #         prob = transition_dict[first_cell, following_cell]
                #         sum_prob += prob
                #         if best_tuple is None:
                #             best_tuple = (first_cell, following_cell)
                #             best_transition_prob = prob
                #         elif prob > best_transition_prob:
                #             best_tuple = (first_cell, following_cell)
                #             best_transition_prob = prob

                seq_transition_dict[long_seq_index_1, long_seq_index_2] = sum_prob

        graph_seq = build_graph_from_transition_dict(seq_transition_dict, n_connected_cell_to_add=2,
                                                     use_longest_path=use_longest_path,
                                                     with_weight=with_weight)
        if debug_mode:
            print(f"organizing graph of sequences")
        seq_indices_list, isolates_seq = find_paths_in_a_graph(graph_seq, shortest_path_on_weight,
                                                               with_weight,
                                                               use_longest_path=use_longest_path,
                                                               debug_mode=debug_mode)
        # creating new cell order
        new_cell_order = []
        cells_to_highlight_colors = []
        cells_to_highlight = []
        span_cells_to_highlight = []
        span_cells_to_highlight_colors = []
        cell_index_so_far = 0
        cell_for_span_index_so_far = 0
        color_index_by_sub_seq = 0
        seq_times_to_color_dict = dict()
        link_seq_color = "white"
        # dict to keep the number of rep of each seq depending on the cells it is composed from
        use_seq_from_graph_to_fill_seq_dict = True
        seq_times_by_seq_cells_dict = dict()

        for color_index, seq_indices in enumerate(seq_indices_list):
            n_cells_in_group_of_seq = 0
            seq_fusion_cells = []
            for seq_index in seq_indices:
                seq = long_seq_list[seq_index]
                n_cells_in_seq = len(seq)
                add_links_to_raster_here = False
                if add_links_to_raster_here:
                    if debug_mode:
                        print(f"new_cell_order.extend {len(seq)}: {seq}")
                    seq_times, seq_dict = get_seq_times_from_raster(spike_nums[seq], min_time_bw_2_spikes,
                                                                    max_time_bw_2_spikes, error_rate=0.3,
                                                                    max_errors_in_a_row=4,
                                                                    cell_indices=seq,
                                                                    min_len_ratio=0.4)  # , min_seq_len=3
                    if use_seq_from_graph_to_fill_seq_dict and (len(seq_times) > 0) and (len(seq) >= 3):
                        seq_times_by_seq_cells_dict[tuple(seq)] = seq_times
                    if debug_mode:
                        print(f"Nb rep seq {len(seq_times)}: {seq_times}")
                    # adding sequences to a dict use to display them in the raster
                    if len(seq_times) > 0:
                        new_seq_indices = np.arange(cell_index_so_far, cell_index_so_far + n_cells_in_seq)
                        for times in seq_times:
                            # keeping the cells that spikes for each sequence of "seq"
                            indices_to_keep = np.where(np.array(times) > -1)[0]
                            cells_to_keep = tuple(new_seq_indices[indices_to_keep])
                            times_to_keep = np.array(times)[indices_to_keep]
                            if cells_to_keep not in seq_times_to_color_dict:
                                seq_times_to_color_dict[cells_to_keep] = []
                            seq_times_to_color_dict[cells_to_keep].append(times_to_keep)
                new_cell_order.extend(seq)
                seq_fusion_cells.extend(seq)
                # n_cells_in_seq += len(seq)
                n_cells_in_group_of_seq += n_cells_in_seq
                cells_to_highlight.extend(np.arange(cell_index_so_far, cell_index_so_far + n_cells_in_seq))
                cell_index_so_far += n_cells_in_seq
                cells_to_highlight_colors.extend([colors[color_index_by_sub_seq % len(colors)]] * n_cells_in_seq)
                color_index_by_sub_seq += 1

            add_links_to_raster_here = True
            if add_links_to_raster_here:
                if debug_mode:
                    print(f"seq_fusion_cells {len(seq_fusion_cells)}")
                seq_times, seq_dict = get_seq_times_from_raster(spike_nums[np.array(seq_fusion_cells)],
                                                      min_time_bw_2_spikes,
                                                      max_time_bw_2_spikes, error_rate=0.3,
                                                      max_errors_in_a_row=4,
                                                      cell_indices=np.array(seq_fusion_cells),
                                                      min_len_ratio=0.4, min_seq_len=3)
                for seq_cells, seq_dict_times in seq_dict.items():
                    if seq_cells not in seq_times_by_seq_cells_dict:
                        seq_times_by_seq_cells_dict[seq_cells] = []
                    seq_times_by_seq_cells_dict[seq_cells].extend(seq_dict_times)
                if debug_mode:
                    print(f"Nb rep seq {len(seq_times)}")
                # adding sequences to a dict use to display them in the raster
                if len(seq_times) > 0:
                    new_seq_indices = np.arange(cell_for_span_index_so_far,
                                                     cell_for_span_index_so_far + n_cells_in_group_of_seq)
                    for times in seq_times:
                        # keeping the cells that spikes for each sequence of "seq"
                        indices_to_keep = np.where(np.array(times) > -1)[0]
                        cells_to_keep = tuple(new_seq_indices[indices_to_keep])
                        times_to_keep = np.array(times)[indices_to_keep]
                        if cells_to_keep not in seq_times_to_color_dict:
                            seq_times_to_color_dict[cells_to_keep] = []
                        seq_times_to_color_dict[cells_to_keep].append(times_to_keep)
            span_cells_to_highlight.extend(np.arange(cell_for_span_index_so_far,
                                                     cell_for_span_index_so_far + n_cells_in_group_of_seq))
            span_cells_to_highlight_colors.extend([colors[color_index % len(colors)]] * n_cells_in_group_of_seq)
            cell_for_span_index_so_far += n_cells_in_group_of_seq

        #     new_cell_order.extend(longest_shortest_path)
        #     graph.remove_nodes_from(longest_shortest_path)
        #     cells_to_highlight.extend(np.arange(cell_index_so_far, cell_index_so_far + len(longest_shortest_path)))
        #     cell_index_so_far += len(longest_shortest_path)
        #     cells_to_highlight_colors.extend([colors[sequence_index % len(colors)]] * len(longest_shortest_path))
        #     sequence_index += 1

        new_cell_order.extend(short_seq_cells)
        new_cell_order.extend(isolates_cell[::-1])

        plot_spikes_raster(spike_nums=spike_nums[np.array(new_cell_order)], param=param,
                           title=f"raster plot ordered with graph",
                           spike_train_format=False,
                           file_name=f"{descr}_raster_plot_ordered_with_graph",
                           y_ticks_labels=new_cell_order,
                           save_raster=True,
                           show_raster=False,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           span_cells_to_highlight=span_cells_to_highlight,
                           span_cells_to_highlight_colors=span_cells_to_highlight_colors,
                           spike_shape='|',
                           spike_shape_size=5,
                           save_formats="pdf")
        n_cells_to_zoom = 150
        plot_spikes_raster(spike_nums=spike_nums[np.array(new_cell_order)][:n_cells_to_zoom], param=param,
                           title=f"raster plot ordered with graph",
                           spike_train_format=False,
                           file_name=f"{descr}_raster_plot_ordered_with_graph_zoom",
                           y_ticks_labels=new_cell_order[:n_cells_to_zoom],
                           save_raster=True,
                           show_raster=False,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           # cells_to_highlight=cells_to_highlight[:n_cells_to_zoom],
                           # cells_to_highlight_colors=cells_to_highlight_colors[:n_cells_to_zoom],
                           # span_cells_to_highlight=span_cells_to_highlight,
                           # span_cells_to_highlight_colors=span_cells_to_highlight_colors,
                           spike_shape='o',
                           spike_shape_size=0.5,
                           # seq_times_to_color_dict=seq_times_to_color_dict,
                           # link_seq_color=link_seq_color,
                           link_seq_line_width=0.5,
                           link_seq_alpha=0.9,
                           save_formats="pdf")

        # dict to keep the number of rep of each seq depending on its size
        seq_stat_dict = dict()
        for seq_cells, times in seq_times_by_seq_cells_dict.items():
            len_seq = len(seq_cells)
            n_rep = len(times)
            if debug_mode:
                print(f"n_rep {n_rep}, times: {times}")
            if len_seq not in seq_stat_dict:
                seq_stat_dict[len_seq] = []
            seq_stat_dict[len_seq].append(n_rep)

        file_name = f'{param.path_results}/significant_sorting_results_{descr}.txt'
        with open(file_name, "w", encoding='UTF-8') as file:
            for n_cells_in_seq, n_rep in seq_stat_dict.items():
                file.write(f"{n_cells_in_seq}:{n_rep}" + '\n')

        save_on_file_seq_detection_results(best_cells_order=new_cell_order,
                                           seq_dict=seq_times_by_seq_cells_dict,
                                           file_name=f"significant_sorting_results_with_timestamps_{descr}.txt",
                                           param=param)
        """
        One idea
        https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.approximation.kcomponents.k_components.html#networkx.algorithms.approximation.kcomponents.k_components
        from networkx.algorithms import approximation as apxa
        G = nx.petersen_graph()
        k_components = apxa.k_components(G)
        use kcomponents to find the subgraphs and then order each subgraph and return the total order
        starting by the longest subgraph

        We could also use https://en.wikipedia.org/wiki/Strongly_connected_component for
        connected graph, but then a node to belong to more than one graph
        """

def save_on_file_seq_detection_results(best_cells_order, seq_dict, file_name, param):
    complete_file_name = f'{param.path_results}/{file_name}'
    with open(complete_file_name, "w", encoding='UTF-8') as file:
        file.write("best_order:")
        for cell_id, cell in enumerate(best_cells_order):
            file.write(f"{cell}")
            if cell_id < (len(best_cells_order) - 1):
                file.write(" ")
        file.write("\n")
        for cells, value in seq_dict.items():
            for cell_id, cell in enumerate(cells):
                file.write(f"{cell}")
                if cell_id < (len(cells) - 1):
                    file.write(" ")
            file.write(f":")
            for time_stamps_id, time_stamps in enumerate(value):
                for t_id, t in enumerate(time_stamps):
                    file.write(f"{t}")
                    if t_id < (len(time_stamps) - 1):
                        file.write(" ")
                if time_stamps_id < (len(value) - 1):
                    file.write("#")
            file.write("\n")
