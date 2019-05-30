import numpy as np
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.param as p_disc_tools_param
import pattern_discovery.tools.misc as p_disc_tools_misc
from pattern_discovery.structures.seq_tree_v1 import Tree
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.display.raster import plot_spikes_raster
import time
import networkx as nx
import networkx.algorithms.dag as dag
from pattern_discovery.graph.force_directed_graphs import plot_graph_using_fa2
from networkx.algorithms.shortest_paths.unweighted import all_pairs_shortest_path
from networkx.algorithms.shortest_paths.weighted import all_pairs_dijkstra


class MarkovParameters(p_disc_tools_param.Parameters):
    def __init__(self, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 max_branches, stop_if_twin, error_rate, no_reverse_seq, spike_rate_weight,
                 path_results=None, time_str=None, bin_size=1,
                 activity_threshold=None, min_n_errors=1):
        """

        :param time_inter_seq: represent the maximum number of times between two elements of a sequences
        :param min_duration_intra_seq: represent the min number of times between two elements of a sequences,
        :param path_results:
        :param time_str:
        :param bin_size:
        :param min_n_errors: represent the minimum number of errors allowed for any sequence length, then added the
        error_rate
        """
        super().__init__(path_results=path_results, time_str=time_str, bin_size=bin_size)
        self.time_inter_seq = time_inter_seq
        self.min_duration_intra_seq = min_duration_intra_seq
        self.min_len_seq = min_len_seq
        self.min_rep_nb = min_rep_nb
        self.error_rate = error_rate
        self.no_reverse_seq = no_reverse_seq
        self.spike_rate_weight = spike_rate_weight
        self.activity_threshold = activity_threshold
        self.min_n_errors = min_n_errors

        # Tree parameters
        self.max_branches = max_branches
        self.stop_if_twin = stop_if_twin
        # making some change for gitkraken test


def build_mle_transition_dict(spike_nums, min_duration_intra_seq, time_inter_seq,
                              sce_times_bool=None, using_avg_dist=True,
                              spike_rate_weight=None, no_reverse_seq=None,
                              try_uniformity_method=False, debug_mode=False):
    """
    Maximum Likelihood estimation,
    don't take into account the fact that if a neuron A fire after a neuron B ,
    then it decreases the probability than B fires after A
    :param spike_nums:
    :param param:
    :param try_uniformity_method: doesn't work so far
    :return:
    """
    print("building Maximum Likelihood estimation transition dict")
    start_time = time.time()
    use_cross_correlation = False
    nb_neurons = len(spike_nums)
    n_times = len(spike_nums[0, :])
    transition_dict = np.zeros((nb_neurons, nb_neurons))
    # give the average distance between consecutive spikes of 2 neurons, in frames
    if using_avg_dist:
        spikes_dist_dict = np.zeros((nb_neurons, nb_neurons))
        # count to make average
        spikes_count_dict = np.zeros((nb_neurons, nb_neurons))
    else:
        spikes_dist_dict = None
        spikes_count_dict = None
    # results obtain from uniform distribution
    uniform_transition_dict = np.zeros((nb_neurons, nb_neurons))
    # so the neuron with the lower spike rates gets the biggest weight in terms of probability
    if spike_rate_weight and (not try_uniformity_method):
        spike_rates = 1 - p_disc_tools_misc.get_spike_rates(spike_nums)
    else:
        spike_rates = np.ones(nb_neurons)

    # we must first introduce a frequency correction
    # using a uniform_spike_nums with uniformly distributed spike trains having the same spike frequency
    uniform_spike_nums = np.zeros((nb_neurons, n_times), dtype="uint8")
    if try_uniformity_method:
        for n, neuron in enumerate(spike_nums):
            times = np.where(neuron)[0]
            nb_times = len(times)
            if nb_times > 0:
                delta_t = (n_times + 1) // nb_times
                new_spike_times = np.arange(0, nb_times, delta_t)
                uniform_spike_nums[n, new_spike_times] = 1

    # a first round to put probabilities up from neurons B that spikes after neuron A
    for neuron_index, neuron_spikes in enumerate(spike_nums):
        if use_cross_correlation:
            for cell in np.arange(nb_neurons):
                if cell == neuron_index:
                    continue
                xy_cov = np.correlate(neuron_spikes, spike_nums[cell], mode="full")
                center_index = len(xy_cov) // 2
                transition_dict[neuron_index, cell] = np.max(xy_cov[(center_index + min_duration_intra_seq):
                                                                    (center_index + time_inter_seq + 1)])
            continue
        # will count how many spikes of each neuron are following the spike of
        for t in np.where(neuron_spikes)[0]:
            # print(f"min_duration_intra_seq {min_duration_intra_seq}")
            t_min = np.max((0, t + min_duration_intra_seq))
            t_max = np.min((t + time_inter_seq, n_times))
            times_to_check = np.arange(t_min, t_max)
            if sce_times_bool is not None:
                # if > 0, means there is a SCE during that interval, and we don't count it
                # if np.sum(sce_times_bool[t:t_max]) > 0:
                #         continue
                # another option is to remove the times of the SCE from the search:
                times_to_check = times_to_check[sce_times_bool[t:t_max]]
                if len(times_to_check) == 0:
                    continue

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
                if using_avg_dist:
                    first_spike_pos = np.where(spike_nums[p, times_to_check])[0][0]
                    first_spike_pos += min_duration_intra_seq
                    spikes_dist_dict[neuron_index, p] += first_spike_pos
                    spikes_count_dict[neuron_index, p] += 1
                if no_reverse_seq:
                    # see to put transition to 0 ??
                    transition_dict[p, neuron_index] = transition_dict[p, neuron_index] - \
                                                       spike_rates[p]
                if try_uniformity_method:
                    uniform_transition_dict[neuron_index, p] = uniform_transition_dict[neuron_index, p] + \
                                                               spike_rates[p]
                    if no_reverse_seq:
                        # see to put transition to 0 ??
                        uniform_transition_dict[p, neuron_index] = uniform_transition_dict[p, neuron_index] - \
                                                                   spike_rates[p]
            # back to one
            spike_nums[neuron_index, actual_neurons_spikes] = 1
        if try_uniformity_method:
            for t in np.where(uniform_spike_nums[neuron_index, :])[0]:
                t_min = np.max((0, t + min_duration_intra_seq))
                t_max = np.min((t + time_inter_seq, n_times))
                times_to_check = np.arange(t_min, t_max)
                if sce_times_bool is not None:
                    # if > 0, means there is a SCE during that interval, and we don't count it
                    # if np.sum(sce_times_bool[t:t_max]) > 0:
                    #         continue

                    # another option is to remove the times of the SCE from the search:
                    times_to_check = times_to_check[sce_times_bool[t:t_max]]
                    if len(times_to_check) == 0:
                        continue

                actual_neurons_spikes = uniform_spike_nums[neuron_index, :] > 0
                # removing the spikes so they are not found in the later search
                spike_nums[neuron_index, actual_neurons_spikes] = 0

                # Retrieve all cells active during the period of time times_to_check
                pos = np.where(uniform_spike_nums[:, times_to_check])[0]
                # pos = np.unique(pos)
                for p in pos:
                    uniform_transition_dict[neuron_index, p] = uniform_transition_dict[neuron_index, p] + \
                                                               spike_rates[p]
                    if no_reverse_seq:
                        # see to put transition to 0 ??
                        uniform_transition_dict[p, neuron_index] = uniform_transition_dict[p, neuron_index] - \
                                                                   spike_rates[p]
                # back to one
                uniform_spike_nums[neuron_index, actual_neurons_spikes] = 1

        transition_dict[neuron_index, neuron_index] = 0
        if try_uniformity_method:
            uniform_transition_dict[neuron_index, neuron_index] = 0

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
    if try_uniformity_method:
        # we divide the values by the uniform one
        uniform_transition_dict[np.where(uniform_transition_dict < 0)] = 0

    keeping_the_nb_of_rep = True
    if not keeping_the_nb_of_rep:
        # we divide for each neuron the sum of the probabilities to get the sum to 1
        for neuron_index in np.arange(nb_neurons):
            if try_uniformity_method:
                # we divide the values by the uniform one
                other_neurons = np.where(uniform_transition_dict[neuron_index, :])[0]
                for o_n in other_neurons:
                    transition_dict[neuron_index, o_n] = transition_dict[neuron_index, o_n] / \
                                                         uniform_transition_dict[neuron_index, o_n]
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
    if debug_mode:
        print(f'median transition: {np.median(transition_dict)}')
        print(f'mean transition: {np.mean(transition_dict)}')
        print(f'std transition: {np.std(transition_dict)}')
        print(f'min transition: {np.min(transition_dict)}')
        print(f'max transition: {np.max(transition_dict)}')

    stop_time = time.time()
    print(f"Maximum Likelihood estimation transition dict built in {np.round(stop_time - start_time, 3)} s")

    if using_avg_dist:
        # averaging
        spikes_count_dict[spikes_count_dict == 0] = 1
        spikes_dist_dict = np.divide(spikes_dist_dict, spikes_count_dict)

    return transition_dict, spikes_dist_dict


# TODO: Print best_seq + info on data
def give_me_stat_on_sorting_seq_results(results_dict, significant_results_dict,
                                        significant_category_dict_by_len,
                                        significant_seq_dict,
                                        significant_category_dict,
                                        best_cells_order,
                                        neurons_sorted, title, param,
                                        use_sce_times_for_pattern_search, n_surrogate,
                                        labels,
                                        extra_file_name="",
                                        results_dict_surrogate=None, neurons_sorted_surrogate=None,
                                        use_only_uniformity_method=False,
                                        use_loss_score_to_keep_the_best_from_tree=
                                        False,
                                        use_ordered_spike_nums_for_surrogate=False,
                                        percentile_threshold=95, keep_the_longest_seq=False):
    """
    Key will be the length of the sequence and value will be a list of int, representing the nb of rep
    of the different lists
    :param results_dict: Contains stats
    :param significant_seq_dict: key is tuple of int representing a cells sequence, value is a list of list representing
    timestamps of the spikes
    :return:
    """
    file_name = f'{param.path_results}/sorting_results{extra_file_name}_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"{title}" + '\n')
        file.write("" + '\n')
        file.write("Parameters" + '\n')
        file.write("" + '\n')
        file.write(f"n_surrogates {n_surrogate}" + '\n')
        file.write(f"error_rate {param.error_rate}" + '\n')
        file.write(f"max_branches {param.max_branches}" + '\n')
        file.write(f"time_inter_seq {param.time_inter_seq}" + '\n')
        file.write(f"min_duration_intra_seq {param.min_duration_intra_seq}" + '\n')
        file.write(f"min_len_seq {param.min_len_seq}" + '\n')
        file.write(f"min_rep_nb {param.min_rep_nb}" + '\n')
        file.write(f"use_sce_times_for_pattern_search {use_sce_times_for_pattern_search}" + '\n')
        file.write(f"use_only_uniformity_method {use_only_uniformity_method}" + '\n')
        file.write(f"use_loss_score_to_keep_the_best_from_tree {use_loss_score_to_keep_the_best_from_tree}" + '\n')
        file.write(f"use_ordered_spike_nums_for_surrogate {use_ordered_spike_nums_for_surrogate}" + '\n')
        file.write(f"keep_the_longest_seq {keep_the_longest_seq}" + '\n')

        file.write("" + '\n')
        min_len = 1000
        max_len = 0
        for key in results_dict.keys():
            min_len = np.min((key, min_len))
            max_len = np.max((key, max_len))
        if results_dict_surrogate is not None:
            for key in results_dict_surrogate.keys():
                min_len = np.min((key, min_len))
                max_len = np.max((key, max_len))

        # key reprensents the length of a seq
        for key in np.arange(min_len, max_len + 1):
            nb_rep_seq = None
            durations = None
            flat_durations = None
            durations_surrogate = None
            flat_durations_surrogate = None
            nb_rep_seq_surrogate = None
            if key in results_dict:
                nb_rep_seq = results_dict[key]["rep"]
                durations = results_dict[key]["duration"]
                flat_durations = [item for sublist in durations for item in sublist]
            if key in results_dict_surrogate:
                nb_rep_seq_surrogate = results_dict_surrogate[key]["rep"]
                durations_surrogate = results_dict_surrogate[key]["duration"]
                flat_durations_surrogate = [item for sublist in durations_surrogate for item in sublist]
            str_to_write = ""
            str_to_write += f"### Length: {key} cells \n"
            real_data_in = False
            if (nb_rep_seq is not None) and (len(nb_rep_seq) > 0):
                real_data_in = True
                str_to_write += f"# Real data (nb seq: {len(nb_rep_seq)}), " \
                    f"repetition: mean {np.round(np.mean(nb_rep_seq), 3)}"
                if np.std(nb_rep_seq) > 0:
                    str_to_write += f", std {np.round(np.std(nb_rep_seq), 3)}"
                str_to_write += f"#, duration: " \
                    f": mean {np.round(np.mean(flat_durations), 3)}"
                if np.std(flat_durations) > 0:
                    str_to_write += f", std {np.round(np.std(flat_durations), 3)}"
            if (nb_rep_seq_surrogate is not None) and (len(nb_rep_seq_surrogate) > 0):
                if real_data_in:
                    str_to_write += f"\n"
                str_to_write += f"# Surrogate (nb seq: {np.round((len(nb_rep_seq_surrogate) / n_surrogate), 4)}), " \
                    f"repetition: " \
                    f"mean {np.round(np.mean(nb_rep_seq_surrogate), 3)}"
                if np.std(nb_rep_seq_surrogate) > 0:
                    str_to_write += f", std {np.round(np.std(nb_rep_seq_surrogate), 3)}"
                str_to_write += f"#, duration: " \
                    f": mean {np.round(np.mean(flat_durations_surrogate), 3)}"
                if np.std(flat_durations_surrogate) > 0:
                    str_to_write += f", std {np.round(np.std(flat_durations_surrogate), 3)}"
            else:
                if not real_data_in:
                    continue
            str_to_write += f"\n"
            if (nb_rep_seq is not None) and (len(nb_rep_seq) > 0):
                if key in significant_results_dict:
                    str_to_write += f"!!!!!!!!! {len(significant_results_dict[key])} significant sequences " \
                        f"of {key} cells, repetition : mean " \
                        f"{np.round(np.mean(significant_results_dict[key]), 2)}, " \
                        f"std {np.round(np.std(significant_results_dict[key]), 2)} " \
                        f"!!!!!!!!!\n"
                    for sig_seq, time_stamps in significant_seq_dict.items():
                        if len(sig_seq) == key:
                            labels_seq = []
                            for cell in sig_seq:
                                labels_seq.append(labels[cell])
                            str_to_write += f"Cat {significant_category_dict[sig_seq]}: " \
                                f"{labels_seq} repeated {len(time_stamps)}\n"
                else:
                    str_to_write += f"No significant sequences of {key} cells\n"

            str_to_write += '\n'
            str_to_write += '\n'
            file.write(f"{str_to_write}")
        file.write("" + '\n')
        file.write("///// Neurons sorted /////" + '\n')
        file.write("" + '\n')

        for index in np.arange(len(neurons_sorted)):
            go_for = False
            if neurons_sorted_surrogate is not None:
                if neurons_sorted_surrogate[index] == 0:
                    pass
                else:
                    go_for = True
            if (not go_for) and neurons_sorted[index] == 0:
                continue
            str_to_write = f"Neuron {index}, x "
            if neurons_sorted_surrogate is not None:
                str_to_write += f"{neurons_sorted_surrogate[index]} / "
            str_to_write += f"{neurons_sorted[index]}"
            if neurons_sorted_surrogate is not None:
                str_to_write += " (surrogate / real data)"
            str_to_write += '\n'
            file.write(f"{str_to_write}")

    file_name = f'{param.path_results}/significant_sorting_results{extra_file_name}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        for key, value in significant_results_dict.items():
            file.write(f"{key}:{value} {significant_category_dict_by_len[key]}" + '\n')

    save_on_file_seq_detection_results(best_cells_order=best_cells_order, seq_dict=significant_seq_dict,
                                       file_name=f"significant_sorting_results_with_timestamps{extra_file_name}.txt",
                                       param=param,
                                       significant_category_dict=significant_seq_dict)


def save_on_file_seq_detection_results(best_cells_order, seq_dict, file_name, param, significant_category_dict=None):
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
            if significant_category_dict is not None:
                file.write("/")
                file.write(f"{significant_category_dict[cells]}")
            file.write("\n")


def bfs(trans_dict, neuron_to_start, param, n_std_for_threshold=0):
    """
    Breadth First Search, build a Tree according to trans_dict, the first node being neuron_to_start
    :param trans_dict:
    :param neuron_to_start:
    :param param:
    :return:
    """
    current_depth = 0
    # determine the maximum nodes of the tree at the deepest level
    max_branches = param.max_branches
    mean_trans_dict = np.mean(trans_dict)
    std_trans_dict = np.std(trans_dict)
    # keep list of neurons to explore, for each depth, when no more current_neuron for a depth are yet to explore
    # then we only keep the one with shortest accumulated metric
    tree_root = Tree(neuron=neuron_to_start, father=None, prob=1, acc=0, n_order=1)
    nodes_to_expend = {0: [tree_root]}
    # keep a sorted list (by acc, the smaller value first) of the nodes (current_neuron) at the current_depth
    nodes_at_depth = dict()
    nodes_at_depth[current_depth] = SortedList()
    nodes_at_depth[current_depth].add(tree_root)
    while True:
        current_node = nodes_to_expend[current_depth][0]
        nodes_to_expend[current_depth] = nodes_to_expend[current_depth][1:]
        current_neuron = current_node.neuron
        trans_dict_copy = np.copy(trans_dict)
        i = 0
        while i < max_branches:
            # will take the nth (n = max_branches) current_neuron with the higher probability to spike after current_neuron
            # for i in np.arange(max_branches):
            next_neuron = np.argmax(trans_dict_copy[current_neuron, :])
            prob = trans_dict_copy[current_neuron, next_neuron]
            trans_dict_copy[current_neuron, next_neuron] = -1
            # if the next best current_neuron probability is 0, then we end the for loop
            threshold_prob = (mean_trans_dict + (n_std_for_threshold * std_trans_dict))
            # threshold_prob = mean_trans_dict
            if prob <= np.max((0, threshold_prob)):  # < (mean_trans_dict + std_trans_dict): #== 0:
                break
            # if prob < mean_trans_dict:
            # if current_neuron already among the elderies, we skip to the next one
            if (current_node.father is not None) and (next_neuron in current_node.parents):
                if param.stop_if_twin:
                    break
                else:
                    continue
            # print(f"prob in loop for: {prob}")
            i += 1
            tree = Tree(neuron=next_neuron, father=current_node, prob=prob,
                        acc=current_node.acc - np.log(prob), n_order=1)
            if (current_depth + 1) not in nodes_at_depth:
                nodes_at_depth[current_depth + 1] = SortedList()
            nodes_at_depth[current_depth + 1].add(tree)
            current_node.add_child(tree)

        # limit the number of neurons for each depth and add the best "current_neuron"
        # in term of accumulated metriccs to
        # the current_node

        # we need to check how many leafs are in the tree at this current_depth
        # and remove the one with the lower score
        # adding to nodes_to_expend only the best nodes

        if len(nodes_to_expend[current_depth]) == 0:
            current_depth += 1
            if current_depth in nodes_at_depth:
                if len(nodes_at_depth[current_depth]) <= max_branches:
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth]
                else:
                    # keeping the best ones
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth][:max_branches]
                    # then removing from the tree the other ones
                    for n in nodes_at_depth[current_depth][max_branches:]:
                        n.disinherit()
            else:
                break

            if current_depth not in nodes_to_expend:
                break
    return tree_root


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
        # removing cells_to_isolate
        if len(cells_to_isolate) > 0:
            connected_cells_sorted_tmp = connected_cells_sorted
            connected_cells_sorted = []
            for cell_connec in connected_cells_sorted_tmp:
                if cell_connec not in cells_to_isolate:
                    connected_cells_sorted.append(cell_connec)
            connected_cells_sorted = np.array(connected_cells_sorted)
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
                cells_with_same_score = connected_cells_sorted[:len(cells_with_same_score)+1]
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

        connected_cells_sorted = connected_cells_sorted[:n_connected_cell_to_add]
        if n_connected_cell_to_add > 1:
            first_cell_transition_score = transition_dict[cell, connected_cells_sorted[0]]
            cells_with_same_score = np.where(transition_dict[cell, connected_cells_sorted[1:]] ==
                                             first_cell_transition_score)[0]
            if len(cells_with_same_score) > 0:
                # we order them by distance
                if spikes_dist_dict is not None:
                    cells_to_sort = connected_cells_sorted[:len(cells_with_same_score)+1]
                    sorted_by_dist = np.argsort(spikes_dist_dict[cell, cells_to_sort])
                    cells_to_sort = cells_to_sort[sorted_by_dist]
                    connected_cells_sorted[:len(cells_with_same_score)+1] = cells_to_sort
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


def find_paths_in_a_graph(graph, shortest_path_on_weight, with_weight, use_longest_path=False):
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
                                print(f"{list(path)}: {weight}")
                            # else both the same length, then we look at the weight
                            elif lowest_weight_among_best_path > weight:
                                lowest_weight_among_best_path = weight
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
                                print(f"{list(path)}: {weight}")
                            # else both the same length, then we look at the weight
                            elif lowest_weight_among_best_path > weight:
                                lowest_weight_among_best_path = weight
                                print(f"{list(path)}: {weight}")

                print(f"longest_shortest_path {len(longest_shortest_path)}: {longest_shortest_path} "
                      f"{lowest_weight_among_best_path}")
        seq_list.append(longest_shortest_path)
        graph.remove_nodes_from(longest_shortest_path)

        if graph.number_of_nodes() == 0:
            break

    return seq_list, isolates_cell


def find_sequences(spike_nums, param, sce_times_bool=None, try_uniformity_method=False,
                   debug_mode=False,
                   use_loss_score_to_keep_the_best=False, ms=None):
    """

    :param spike_nums:
    :param param:
    :param no_print:
    :return: return dict_by_len_seq, see comment in the code
    """
    time_inter_seq = param.time_inter_seq
    min_len_seq = param.min_len_seq

    # Max len in term of repetition of the max len seq in terms of neuron
    max_rep_non_prob = 0
    max_len_non_prob = 0
    # a dict of dict, with each key of the first dict representing the length
    # of the sequences. The 2nd dict will have as key sequences of neurons and as value the time of the neurons spikes
    dict_by_len_seq = SortedDict()

    # print(f'nb spikes neuron 1: {len(np.where(spike_nums[1,:])[0])}')

    # spike_nums_backup = spike_nums
    spike_nums = np.copy(spike_nums)
    # spike_nums = spike_nums[:, :2000]
    nb_neurons = len(spike_nums)

    transition_dict, spikes_dist_dict = build_mle_transition_dict(spike_nums=spike_nums,
                                                min_duration_intra_seq=param.min_duration_intra_seq,
                                                time_inter_seq=param.time_inter_seq,
                                                try_uniformity_method=try_uniformity_method,
                                                debug_mode=debug_mode,
                                                spike_rate_weight=param.spike_rate_weight,
                                                no_reverse_seq=param.no_reverse_seq,
                                                sce_times_bool=sce_times_bool)
    transition_dict_2_nd_order = None
    try_with_2nd_order = True
    if try_with_2nd_order:
        print("Building 2nd order transition dict")
        transition_dict_2_nd_order, spikes_dist_dict_2_nd_order = build_mle_transition_dict(spike_nums=spike_nums,
                                                               min_duration_intra_seq=param.min_duration_intra_seq * 2,
                                                               time_inter_seq=param.time_inter_seq * 2,
                                                               try_uniformity_method=try_uniformity_method,
                                                               debug_mode=debug_mode,
                                                               spike_rate_weight=param.spike_rate_weight,
                                                               no_reverse_seq=param.no_reverse_seq,
                                                               sce_times_bool=sce_times_bool)

    # print(f"transition_dict {transition_dict}")
    # len of nb_neurons, each element is a dictionary with each key represent a common seq (neurons tuple, first neurons
    # being the index of the list)
    # and each values represent a list of list of times
    list_dict_result = []

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
        n_connected_cell_to_add = 5
        if use_graph_from_connectity_ms and (ms is not None):
            print('use_graph_from_connectity_ms')
            ms.detect_n_in_n_out()
            graph = ms.spike_struct.graph_out
        else:
            # cells_to_isolate = None
            # removing from the graph cells that fires the most and the less
            spike_count_by_cell = np.sum(spike_nums, axis=1)
            low_spike_count_threshold = np.percentile(spike_count_by_cell, 20)
            high_spike_count_threshold = np.percentile(spike_count_by_cell, 95)
            cells_to_isolate = np.where(spike_count_by_cell < low_spike_count_threshold)[0]
            cells_to_isolate = np.concatenate((cells_to_isolate,
                                               np.where(spike_count_by_cell > high_spike_count_threshold)[0]))
            # the idea is to build a graph based on top n connection from the transition_dict
            # directed graph
            graph = build_graph_from_transition_dict(transition_dict, n_connected_cell_to_add,
                                                     use_longest_path=use_longest_path,
                                                     with_weight=with_weight, cells_to_isolate=cells_to_isolate,
                                                     transition_dict_2_nd_order=transition_dict_2_nd_order,
                                                     spikes_dist_dict=spikes_dist_dict,
                                                     min_rep_nb=param.min_rep_nb)

        if plot_graph:
            plot_graph_using_fa2(graph=graph, file_name=f"graph_seq",
                                 title=f"graph_seq",
                                 param=param, iterations=15000, save_raster=True, with_labels=False,
                                 save_formats="pdf", show_plot=False)
        # cycles = nx.simple_cycles(graph)
        # print(f"len(cycles) {len(list(cycles))}")
        print(f"dag.is_directed_acyclic_graph(graph) {dag.is_directed_acyclic_graph(graph)}")

        seq_list, isolates_cell = find_paths_in_a_graph(graph, shortest_path_on_weight, with_weight,
                                                        use_longest_path=False)

        # we have a list of seq that we want to concatenate according to the score of transition between
        # the first and last cell in transition dict
        # first we keep aside the seq that are composed of less than 3 cells
        short_seq_cells = []
        long_seq_list = []
        for seq in seq_list:
            if len(seq) <= 2:
                # organize those ones according to transition dict
                short_seq_cells.extend(seq)
            else:
                print(f"long_seq_list.append {seq}")
                long_seq_list.append(seq)
        n_long_seq = len(long_seq_list)
        seq_transition_dict = np.zeros((n_long_seq, n_long_seq))
        for long_seq_index_1, long_seq_1 in enumerate(long_seq_list):
            for long_seq_index_2, long_seq_2 in enumerate(long_seq_list):
                if long_seq_index_1 == long_seq_index_2:
                    continue
                first_cells = long_seq_1[-2:]
                following_cells = long_seq_2[:2]
                best_tuple = None
                best_transition_prob = 0
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
        print(f"organizing graph of sequences")
        seq_indices_list, isolates_seq = find_paths_in_a_graph(graph_seq, shortest_path_on_weight, with_weight,
                                                               use_longest_path=use_longest_path)
        # creating new cell order
        new_cell_order = []
        cells_to_highlight_colors = []
        cells_to_highlight = []
        cell_index_so_far = 0
        color_index_by_sub_seq = 0
        for color_index, seq_indices in enumerate(seq_indices_list):
            n_cells_in_seq = 0
            for seq_index in seq_indices:
                seq = long_seq_list[seq_index]
                print(f"new_cell_order.extend {seq}")
                new_cell_order.extend(seq)
                # n_cells_in_seq += len(seq)
                n_cells_in_seq = len(seq)
                cells_to_highlight.extend(np.arange(cell_index_so_far, cell_index_so_far + n_cells_in_seq))
                cell_index_so_far += n_cells_in_seq
                cells_to_highlight_colors.extend([colors[color_index_by_sub_seq % len(colors)]] * n_cells_in_seq)
                color_index_by_sub_seq += 1

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
                           file_name=f"raste_plot_ordered_with_graph",
                           y_ticks_labels=new_cell_order,
                           save_raster=True,
                           show_raster=False,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           spike_shape='|',
                           spike_shape_size=5,
                           save_formats="pdf")
        n_cells_to_zoom = 150
        plot_spikes_raster(spike_nums=spike_nums[np.array(new_cell_order)][:n_cells_to_zoom], param=param,
                           title=f"raster plot ordered with graph",
                           spike_train_format=False,
                           file_name=f"raste_plot_ordered_with_graph_zoom",
                           y_ticks_labels=new_cell_order[:n_cells_to_zoom],
                           save_raster=True,
                           show_raster=False,
                           show_sum_spikes_as_percentage=True,
                           plot_with_amplitude=False,
                           cells_to_highlight=cells_to_highlight[:n_cells_to_zoom],
                           cells_to_highlight_colors=cells_to_highlight_colors[:n_cells_to_zoom],
                           spike_shape='o',
                           spike_shape_size=1,
                           save_formats="pdf")

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
        raise Exception("NOT TODAY")

    # key: neuron as integer, value: list of neurons being the longest probable sequence
    max_seq_dict = dict()
    # Start to look for real sequences in spike_nums
    for cell, neuron_spikes in enumerate(spike_nums):
        # if the cell has no spikes, then no sequence start by this cell
        if np.sum(neuron_spikes) == 0:
            max_seq_dict[cell] = [cell]
            continue
        start_time = time.time()
        print(f"Looking for seq starting by cell {cell}")
        tree = bfs(trans_dict=transition_dict, neuron_to_start=cell, param=param)
        # from tree to list of list
        sequences = tree.get_seq_lists()
        if debug_mode:
            print(f'Nb probabilities seq for neuron {cell}: {len(sequences)}')
        # seq = find_seq(trans_dict=transition_dict, neuron_to_start=y, threshold=threshold_prob)
        seq_dict_result = dict()
        # key is a tuple of neurons (int) representing a sequence, value is a set of tuple of times (int) representing
        # the times at which the seq is repeating
        set_seq_dict_result = dict()
        # error_by_seq_dict a dict indicating for each neuron seq instance how many errors was
        # found
        # not correlated to the times of each instance, but allow to do the min and the mean
        error_by_seq_dict = dict()
        # Give the max length among the sequence computed as the most probable
        max_len_prob_seq = 0
        if use_loss_score_to_keep_the_best:
            best_loss_score = 1
            best_seq = None
            for seq in sequences:
                new_order = np.zeros(nb_neurons, dtype="uint16")
                new_order[:len(seq)] = seq

                non_ordered_neurons = np.setdiff1d(np.arange(nb_neurons),
                                                   seq)
                if len(non_ordered_neurons) > 0:
                    ordered_neurons = order_cells_from_seq_dict(seq_dict=set_seq_dict_result,
                                                                non_ordered_neurons=non_ordered_neurons,
                                                                param=param, debug_mode=False)
                    new_order[len(seq):] = ordered_neurons
                    # new_order[len(seq):] = non_ordered_neurons
                    # new_order[len(seq):] = not_ordered_neurons
                tmp_spike_nums = spike_nums[new_order, :]
                loss_score = loss_function_with_sliding_window(spike_nums=tmp_spike_nums[::-1, :],
                                                               time_inter_seq=param.time_inter_seq,
                                                               min_duration_intra_seq=param.min_duration_intra_seq)
                if loss_score < best_loss_score:
                    best_seq = seq
                    best_loss_score = loss_score
            max_seq_dict[cell] = best_seq

        for seq in sequences:
            # print(f"Neuron {y}, tree seq: {seq}")
            # look for each spike of this neuron, if a seq is found on the following spikes of other neurons
            for t in np.where(neuron_spikes)[0]:
                # look if the sequence is right
                index = t + param.min_duration_intra_seq
                if index < 0:
                    index = 0
                elif t >= len(spike_nums[0, :]):
                    index = len(spike_nums[0, :]) - 1
                # if not seq_can_be_on_same_time:
                #     if t >= (len(spike_nums[0, :]) - 1):
                #         break
                #     else:
                #         index = t + 1
                # else:
                #     if t > 1:
                #         index = t-2
                #     else:
                #         index = t
                # keep neurons numbers
                neurons_sequences = []
                times_sequences = []
                # How many times which outpass the fact that one neuron was missing
                nb_error = 0
                # Check if the last neuron added to the sequence is an error
                # the sequence must finish by a predicted neuron
                last_is_error = np.zeros(2)

                # for each neuron of the most probable sequence
                for nb_s, n_seq in enumerate(seq):
                    new_time_inter_seq = time_inter_seq
                    if last_is_error.any():
                        to_add = (time_inter_seq * 2)
                        if last_is_error.all():
                            to_add *= 2
                        if (index + to_add) < len(spike_nums[n_seq, :]):
                            new_time_inter_seq = to_add
                    # If too many neurons actives during the following time_inter_seq
                    # then we stop here
                    # TODO: Keep it ?
                    take_sce_into_account = False
                    if take_sce_into_account:
                        if np.sum(spike_nums[:, index:(index + new_time_inter_seq)]) > \
                                (0.0128 * new_time_inter_seq * len(spike_nums)):
                            print(f'np.sum(spike_nums[:, index:(index + new_time_inter_seq)]) > '
                                  f'(0.0128*len(spike_nums))')
                            break
                    # look if the neuron n_seq during the next time_inter_seq time is activated
                    if np.any(spike_nums[n_seq, index:(index + new_time_inter_seq)]):
                        time_index_list = np.where(spike_nums[n_seq, index:(index + new_time_inter_seq)])[0]
                        neurons_sequences.append(n_seq)
                        # print(f'time_index {time_index}')
                        next_spike_time = np.min(time_index_list)
                        # print(f'index {index}, next_spike_time {next_spike_time}')
                        times_sequences.append(index + next_spike_time)
                        index += next_spike_time
                        last_is_error = np.zeros(2)
                        if (index + time_inter_seq) >= (len(spike_nums[n_seq, :]) - 1):
                            # if nb_s >= (min_len_seq - 1):
                            #     neurons_sequences = seq[:nb_s + 1]
                            break
                    else:
                        # the last neuron should be a neuron from the sequence, we just skip one, two neurons can't be
                        # skip one after the other one
                        if (nb_error < (int(len(neurons_sequences) * param.error_rate) + param.min_n_errors)) and \
                                (not last_is_error.all()):
                            nb_error += 1
                            if last_is_error.any():
                                last_is_error[1] = 1
                            else:
                                last_is_error[0] = 1
                            # appending n_seq even so it'n_seq not found
                            neurons_sequences.append(n_seq)
                            times_sequences.append(index)
                        else:
                            break
                if last_is_error.any():
                    to_remove = int(-1 * np.sum(last_is_error))
                    nb_error -= np.sum(last_is_error)
                    # print(f'to_remove {to_remove}')
                    if len(neurons_sequences) > 0:
                        neurons_sequences = neurons_sequences[:to_remove]
                        times_sequences = times_sequences[:to_remove]
                # saving the sequence only if its duration has a mimimum length and repeat a minimum of times
                if (len(neurons_sequences) >= param.min_len_seq) and (len(times_sequences) >= param.min_rep_nb):
                    time_neurons_seq = tuple(times_sequences)
                    neurons_sequences = tuple(neurons_sequences)
                    # print(f"neuron {y} selected seq {neurons_sequences}")
                    len_seq = len(neurons_sequences)

                    add_seq = True
                    if neurons_sequences not in set_seq_dict_result:
                        set_seq_dict_result[neurons_sequences] = set()
                        set_seq_dict_result[neurons_sequences].add(time_neurons_seq)
                        # we need to check if some time sequences don't intersect too much with this one,
                        # otherwise we don't add the seq to the dictionnary
                    else:
                        for t_seq in set_seq_dict_result[neurons_sequences]:
                            perc_threshold = 0.3
                            inter = np.intersect1d(np.array(time_neurons_seq), np.array(t_seq))
                            if len(inter) > (0.2 * len(t_seq)):
                                add_seq = False
                                break
                        if add_seq:
                            set_seq_dict_result[neurons_sequences].add(time_neurons_seq)

                    if add_seq:
                        # error_by_seq_dict a dict indicating for each neuron seq instance how many errors was
                        # found
                        # not correlated to the times of each instance, but allow to do the min and the mean
                        if neurons_sequences not in error_by_seq_dict:
                            error_by_seq_dict[neurons_sequences] = []
                        error_by_seq_dict[neurons_sequences].append(nb_error)
                        if not use_loss_score_to_keep_the_best:
                            if len(seq) > max_len_prob_seq:
                                # Keep the whole sequences, not just the segment that was actually on the plot
                                # But only if at least one part of sequence was found on the plot, and a minimum
                                # of time (set into param)
                                max_len_prob_seq = len(seq)
                                max_seq_dict[cell] = seq
                        # dict_by_len_seq will be return
                        if len_seq not in dict_by_len_seq:
                            dict_by_len_seq[len_seq] = dict()
                        if neurons_sequences not in dict_by_len_seq[len(neurons_sequences)]:
                            dict_by_len_seq[len_seq][neurons_sequences] = set()
                        dict_by_len_seq[len_seq][neurons_sequences].add(time_neurons_seq)

                        if time_neurons_seq not in seq_dict_result:
                            seq_dict_result[time_neurons_seq] = neurons_sequences

                    # neurons_seq.append(neurons_sequences)
                    # time_neurons_seq.append(times_sequences)

        if debug_mode:
            print(f'Len max for neuron {cell} prob seq: {max_len_prob_seq}')
        # print_save(f'Nb seq for neuron {y}: {len(seq_dict_result)}', file, no_print=True, to_write=write_on_file)
        # if more than one sequence for one given neuron, we draw them
        if len(seq_dict_result) > 1:
            list_dict_result.append(set_seq_dict_result)
            # TODO: see if it's a good idea
            # removing seq for which no instance have less than 2 errors comparing to probabilistic seq
            removing_seq_with_errors = False
            if removing_seq_with_errors:
                for k, v in error_by_seq_dict.items():
                    if np.min(v) > 1:
                        # print(f'min nb errors: {np.min(v)}')
                        set_seq_dict_result.pop(k, None)
                        dict_by_len_seq[len(k)].pop(k, None)
            # remove intersecting seq and seq repeting less than
            # TODO: change it to work on dict_by_len_seq
            removing_intersect_seq(set_seq_dict_result, min_len_seq, param.min_rep_nb, dict_by_len_seq)
            # for neurons_seq, times_seq_set in set_seq_dict_result.items():
            #     # if (len(times_seq_set) > 2) and (len(neurons_seq) >= param.min_len_seq_first_tour):
            #     if (len(times_seq_set) > max_rep_non_prob) and \
            #             (len(neurons_seq) >= np.max(param.min_len_seq_first_tour)):
            #         max_rep_non_prob = len(times_seq_set)
            #         max_len_non_prob = len(neurons_seq)
        stop_time = time.time()
        print(f"Looked for seq starting by cell {cell} in {np.round(stop_time - start_time, 3)} s")
    # max_rep_non_prob is the maximum rep of a sequence found in spike_nums
    return list_dict_result, dict_by_len_seq, max_seq_dict


def find_sequences_in_ordered_spike_nums(spike_nums, param, debug_mode=False):
    """
    Find sequence in spike_nums starting from cell 0, with respect of param (such as len_seq, rep_seq etc..)
    :param spike_nums:
    :param param:
    :return:
    """
    seq_dict = dict()
    n_cells = len(spike_nums)
    n_times = len(spike_nums[0, :])
    # used to merge seq
    intersec_coeff = 0.2

    for cell_id, cell_times in enumerate(spike_nums):
        if (n_cells - 1 - cell_id) < param.min_len_seq:
            break
        current_seq_dict = dict()
        cell_spikes = p_disc_tools_misc.get_continous_time_periods(cell_times)
        # cell_spikes = np.where(cell_times)[0]
        for spike_id, spike_times in enumerate(cell_spikes):
            # to avoid the same spike if using rasterdur
            # no need anymore thanks to get_continous_time_periods
            # if spike_id > 0:
            #     if spike_time == (cell_spikes[spike_id - 1] + 1):
            #         continue
            last_spike_time = spike_times[0]
            nb_errors = 0
            errors_index = []
            index_seq = 1
            current_seq_cells = [cell_id]
            current_seq_times = [spike_times[0]]
            time_inter_seq = param.time_inter_seq
            min_duration_intra_seq = param.min_duration_intra_seq
            for next_cell_id in np.arange(cell_id + 1, n_cells):
                min_time = np.max((0, last_spike_time + min_duration_intra_seq))
                max_time = np.min((n_times, last_spike_time + time_inter_seq))

                spikes_next_cell = np.where(spike_nums[next_cell_id, min_time:max_time])[0]
                if len(spikes_next_cell) > 0:
                    current_seq_times.append(spikes_next_cell[0] + min_time)
                    last_spike_time = spikes_next_cell[0] + min_time
                    current_seq_cells.append(next_cell_id)
                    time_inter_seq = param.time_inter_seq
                    min_duration_intra_seq = param.min_duration_intra_seq
                else:
                    if nb_errors < (int(len(current_seq_cells) * param.error_rate) + param.min_n_errors):
                        nb_errors += 1
                        errors_index.append(index_seq)
                        current_seq_cells.append(next_cell_id)
                        # put a fake time, where no spike exist for this cell
                        current_seq_times.append(min_time)
                        time_inter_seq += (param.time_inter_seq // 2)
                        min_duration_intra_seq += (param.min_duration_intra_seq // 2)

                    else:
                        break

                index_seq += 1

            # first if errors have been added at the end, we remove them
            while len(errors_index) > 0:
                # print(f"errors_index {errors_index}, len(current_seq_cells) {len(current_seq_cells)}")
                if errors_index[-1] == (len(current_seq_cells) - 1):
                    current_seq_cells = current_seq_cells[:-1]
                    current_seq_times = current_seq_times[:-1]
                    errors_index = errors_index[:-1]
                else:
                    break

            # if too many errors comparing to the length of the seq, we don't keep it
            # not used normally
            if len(errors_index) > (int(len(current_seq_cells) * param.error_rate) + param.min_n_errors):
                print(f"len(errors_index) > int(len(current_seq_cells) * param.error_rate) "
                      f"len(errors_index) {len(errors_index)}, "
                      f"int(len(current_seq_cells) * param.error_rate) "
                      f"{int(len(current_seq_cells) * param.error_rate)}")
                continue

            # then we check if the seq has the min length
            if len(current_seq_cells) >= param.min_len_seq:
                # print(f"current_seq_cells {current_seq_cells}")
                current_seq_cells_backup = current_seq_cells[:]
                current_seq_times_backup = current_seq_times[:]
                nb_errors_to_add = int(len(current_seq_cells) * param.error_rate) + param.min_n_errors - \
                                   len(errors_index)

                # new version
                first_cell = current_seq_cells_backup[0]

                current_seq_cells = current_seq_cells_backup
                current_seq_times = current_seq_times_backup
                # nb_errors_to_add -= 1

                tuple_seq = tuple(current_seq_cells)

                # we add errors before only in order to see if it match sequences already added with previous
                # cells
                # if (tuple_seq not in current_seq_dict) and nb_errors_to_add >= 0 and ((first_cell - 1) >= 0):
                #     continue

                if tuple_seq not in current_seq_dict:
                    # first we want to check if a seq already in current_seq_dict that will be longer could be
                    # considered the same
                    seq_added = False
                    seq_to_remove = []
                    for seq_in, seq_times_in in current_seq_dict.items():
                        if len(seq_in) > len(tuple_seq):
                            cells_diff = np.setdiff1d(seq_in, tuple_seq)
                            if len(cells_diff) <= len(tuple_seq) * intersec_coeff:
                                current_seq_times = current_seq_times + ([current_seq_times[-1]] * len(cells_diff))
                                # print(f"len(current_seq_times) {len(current_seq_times)}, len(seq_in) {len(seq_in)}")
                                current_seq_dict[seq_in].append(current_seq_times)
                                seq_added = True
                                break
                        elif len(seq_in) <= len(tuple_seq):
                            cells_diff = np.setdiff1d(tuple_seq, seq_in)
                            if len(cells_diff) <= len(seq_in) * intersec_coeff:
                                seq_to_remove.append(seq_in)
                                new_seq_times_in = []
                                for seq_times in seq_times_in:
                                    new_seq_times_in.append(list(seq_times) + ([seq_times[-1]] * len(cells_diff)))
                                # print(f"new_seq_times_in {new_seq_times_in}")
                                current_seq_dict[tuple_seq] = [current_seq_times]
                                current_seq_dict[tuple_seq].extend(new_seq_times_in)
                                seq_added = True
                                break
                    if seq_added:
                        if len(seq_to_remove) > 0:
                            for key in seq_to_remove:
                                if key in current_seq_dict:
                                    del current_seq_dict[key]
                        continue

                    current_seq_dict[tuple_seq] = []
                    current_seq_dict[tuple_seq].append(current_seq_times)
                else:
                    # first we check that the times of the new seq are no intersect with other one
                    ok_to_add_it = True
                    for times_seq_already_in in current_seq_dict[tuple_seq]:
                        for time_id, time_value in enumerate(times_seq_already_in):
                            # we can't use intersect of setdiff
                            # print(f"time_value {time_value}, current_seq_times[time_id] {current_seq_times[time_id]}")
                            if time_value == current_seq_times[time_id]:
                                ok_to_add_it = False
                                break
                        if not ok_to_add_it:
                            break
                    # print(f"ok_to_add_it {ok_to_add_it}")
                    if ok_to_add_it:
                        # print(f"nb_errors_to_add {nb_errors_to_add}, "
                        #       f"ok_to_add_it {current_seq_cells} / {current_seq_times}")
                        current_seq_dict[tuple_seq].append(current_seq_times)

                # while not_added and (nb_errors_to_add >= 0):
                #     first_cell = current_seq_cells_backup[0]
                #     if (nb_errors_to_add > 0) and ((first_cell - 1) >= 0):
                #         first_cell_time = current_seq_times_backup[0]
                #         current_seq_cells = [first_cell - 1] + current_seq_cells_backup
                #         current_seq_times = [first_cell_time] + current_seq_times_backup
                #     else:
                #         current_seq_cells = current_seq_cells_backup
                #         current_seq_times = current_seq_times_backup
                #     nb_errors_to_add -= 1
                #
                #     tuple_seq = tuple(current_seq_cells)
                #
                #     # we add errors before only in order to see if it match sequences already added with previous
                #     # cells
                #     if (tuple_seq not in current_seq_dict) and nb_errors_to_add >= 0 and ((first_cell - 1) >= 0):
                #         continue
                #
                #     if tuple_seq not in current_seq_dict:
                #         current_seq_dict[tuple_seq] = []
                #         current_seq_dict[tuple_seq].append(current_seq_times)
                #         not_added = False
                #     else:
                #         # first we check that the times of the new seq are no intersect with other one
                #         ok_to_add_it = True
                #         for times_seq_already_in in current_seq_dict[tuple_seq]:
                #             for time_id, time_value in enumerate(times_seq_already_in):
                #                 # we can't use intersect of setdiff
                #                 if time_value == current_seq_times[time_id]:
                #                     ok_to_add_it = False
                #                     break
                #             if not ok_to_add_it:
                #                 break
                #         # print(f"ok_to_add_it {ok_to_add_it}")
                #         if ok_to_add_it:
                #             # print(f"nb_errors_to_add {nb_errors_to_add}, "
                #             #       f"ok_to_add_it {current_seq_cells} / {current_seq_times}")
                #             current_seq_dict[tuple_seq].append(current_seq_times)
                #             not_added = False
        # print(f"current_seq_dict.keys() {list(current_seq_dict.keys())}")
        # seq_to_remove = []
        # seq_to_remove_from_valid_seq = []
        keys_current_seq_dict = list(current_seq_dict.keys())
        index_key = 0
        # if cell_id <= 5:
        #     print("")
        #     print(f"###### cell_id {cell_id} ##########")
        #     print(f"keys_current_seq_dict {len(keys_current_seq_dict)}:{keys_current_seq_dict}")
        #     print(f"nb_rep current: {[len(x) for x in (list(current_seq_dict.values()))]}")
        #     print(f"keys seq_dict {len(list(seq_dict.keys()))}: {list(seq_dict.keys())}")
        #     print(f"nb_rep {[len(x) for x in (list(seq_dict.values()))]}")

        already_checked_seq = dict()
        # for key in keys_current_seq_dict:
        while True:
            go_out = True
            for key in current_seq_dict.keys():
                if key not in already_checked_seq:
                    already_checked_seq[key] = 1
                    go_out = False
                    break
            if go_out:
                break

            value = current_seq_dict[key]
            # if len(value) < param.min_rep_nb:
            #     seq_to_remove.append(key)
            #     continue
            # check if there is no intersection with seq already in seq_dict
            valid_seqs = list(seq_dict.keys())
            for valid_seq in valid_seqs:
                valid_times = seq_dict[valid_seq]
                if len(key) <= len(valid_seq):
                    long_dict = seq_dict
                    short_dict = current_seq_dict
                    long_seq = valid_seq
                    short_seq = key
                    long_times = valid_times
                    short_times = value
                else:
                    long_dict = current_seq_dict
                    short_dict = seq_dict
                    long_seq = key
                    short_seq = valid_seq
                    long_times = value
                    short_times = valid_times

                unique_cells = np.setdiff1d(short_seq, long_seq)
                to_remove = False
                if len(unique_cells) == 0:
                    # mean that short_seq is included in long_seq
                    # we want to see if the shorter seq is always at the same time of the longer one
                    # if not, then we keep it
                    if not is_seq_independant(times_short_seq=short_times, times_long_seq=long_times):
                        # seq_to_remove.append(short_seq)
                        del short_dict[short_seq]
                        break
                        to_remove = True
                    if not to_remove:
                        # if cell_id <= 5:
                        #     print(f"step1")
                        #     print(f"short_seq :{short_seq}")
                        #     print(f"long_seq: {long_seq}")
                        # if two seq have the same end, we increase the size of the short one to make them one
                        if (short_seq[-1] == long_seq[-1]) and \
                                ((len(long_seq) - len(short_seq)) <= len(short_seq) * intersec_coeff):
                            # print("new condition")
                            new_seq_times = []
                            for short_seq_times in short_times:
                                alread_in = False
                                # Add this short_seq_times if not intersecting will another seq_time
                                for long_time in long_times:
                                    if (short_seq_times[0] >= long_time[0]) and (short_seq_times[-1] <= long_time[-1]):
                                        alread_in = True
                                        break
                                    if len(np.intersect1d(np.array(short_seq_times), np.array(long_time))):
                                        alread_in = True
                                        break
                                if not alread_in:
                                    short_seq_times = ([short_seq_times[0]] *
                                                       (len(long_seq) - len(short_seq))) + short_seq_times
                                    new_seq_times.append(short_seq_times)
                            if len(new_seq_times) > 0:
                                long_dict[long_seq].extend(new_seq_times)
                            del short_dict[short_seq]
                            break

                        cells_diff = np.setdiff1d(long_seq, short_seq)
                        # print(f"cells_diff {cells_diff}")
                        if len(cells_diff) <= (len(short_seq) * intersec_coeff):
                            index_beg = np.where(np.array(long_seq) == short_seq[0])[0]
                            if len(index_beg) > 0:
                                # print(f"len(index_beg) > 0 {len(index_beg) > 0}")
                                # print(f"long_seq {long_seq}, short_seq {short_seq}")
                                index_beg = index_beg[0]
                                to_add_at_the_end = len(cells_diff) - index_beg
                                new_seq_times = []
                                for short_seq_times in short_times:
                                    alread_in = False
                                    # Add this short_seq_times if not intersecting will another seq_time
                                    for long_time in long_times:
                                        if (short_seq_times[0] >= long_time[0]) and (
                                                short_seq_times[-1] <= long_time[-1]):
                                            alread_in = True
                                            break
                                        if len(np.intersect1d(np.array(short_seq_times), np.array(long_time))):
                                            alread_in = True
                                            break
                                    if not alread_in:
                                        short_seq_times = ([short_seq_times[0]] * index_beg) + short_seq_times
                                        if to_add_at_the_end > 0:
                                            short_seq_times = short_seq_times + (
                                                    [short_seq_times[-1]] * to_add_at_the_end)
                                        new_seq_times.append(short_seq_times)
                                if len(new_seq_times) > 0:
                                    long_dict[long_seq].extend(new_seq_times)
                                del short_dict[short_seq]
                                break
                else:
                    if len(unique_cells) <= len(short_seq) * intersec_coeff:
                        new_seq = []
                        # to_add_beg_long = 0
                        # to_add_beg_short = 0
                        # to_add_end_long = 0
                        # to_add_end_short = 0
                        # then we merge both
                        diff_beg_long_short = np.abs(long_seq[0] - short_seq[0])
                        diff_end_long_short = np.abs(long_seq[-1] - short_seq[-1])
                        if short_seq[0] <= long_seq[0]:
                            new_seq.extend(list(short_seq[:diff_beg_long_short]))
                            new_seq.extend(list(long_seq))
                            to_add_beg_long = diff_beg_long_short
                            to_add_end_long = 0
                            to_add_beg_short = 0
                            to_add_end_short = diff_end_long_short
                        else:
                            new_seq.extend(list(long_seq))
                            new_seq.extend(list(short_seq[-diff_end_long_short:]))
                            to_add_beg_long = 0
                            to_add_end_long = diff_end_long_short
                            to_add_beg_short = diff_beg_long_short
                            to_add_end_short = 0

                        new_seq_times = []
                        for short_seq_times in short_times:
                            alread_in = False
                            # Add this short_seq_times if not intersecting will another seq_time
                            for long_seq_time in long_times:
                                if (short_seq_times[0] >= long_seq_time[0]) and (
                                        short_seq_times[-1] <= long_seq_time[-1]):
                                    alread_in = True
                                    break
                                if len(np.intersect1d(np.array(short_seq_times), np.array(long_seq_time))):
                                    alread_in = True
                                    break
                            if not alread_in:
                                short_seq_times = ([short_seq_times[0]] * to_add_beg_short) + short_seq_times
                                if to_add_end_short > 0:
                                    short_seq_times = short_seq_times + (
                                            [short_seq_times[-1]] * to_add_end_short)
                                new_seq_times.append(short_seq_times)
                        for long_seq_time in long_times:
                            long_seq_time = ([long_seq_time[0]] * to_add_beg_long) + long_seq_time
                            if to_add_end_long > 0:
                                long_seq_time = long_seq_time + (
                                        [long_seq_time[-1]] * to_add_end_long)
                            new_seq_times.append(long_seq_time)
                        del short_dict[short_seq]
                        del long_dict[long_seq]
                        # adding it to current_seq_dict_dict so we can check the new seq with seq already in seq_dict
                        # seq_dict[tuple(new_seq)] = new_seq_times
                        current_seq_dict[tuple(new_seq)] = new_seq_times
                        break
            # keys_current_seq_dict = list(current_seq_dict.keys())
            # index_key += 1

        # for key in seq_to_remove:
        #     if key in current_seq_dict:
        #         del current_seq_dict[key]

        # for key in seq_to_remove_from_valid_seq:
        #     if key in seq_dict:
        #         del seq_dict[key]

        seq_dict.update(current_seq_dict)

        # if cell_id <= 5:
        #     print(f"## keys current_seq_dict {len(list(current_seq_dict.keys()))}: {list(current_seq_dict.keys())}")
        #     print(f"## keys seq_dict {len(list(seq_dict.keys()))}: {list(seq_dict.keys())}")
        #     print(f"nb_rep {[len(x) for x in (list(seq_dict.values()))]}")

        # if cell_id > 5:
        #     raise Exception("> 5")

    # we need to filter the dict to remove seq that don't repeat enough
    seq_to_remove_from_valid_seq = []
    for key, value in seq_dict.items():
        if len(value) < param.min_rep_nb:
            seq_to_remove_from_valid_seq.append(key)
            continue

    for key in seq_to_remove_from_valid_seq:
        if key in seq_dict:
            del seq_dict[key]

    if debug_mode:
        print("")
        print("seq_dict")
        for key, times in seq_dict.items():
            print(f"## key rep {len(times)} len {len(key)}: {key}")

    return seq_dict


def is_seq_independant(times_short_seq, times_long_seq):
    """

    :param times_short_seq: list of list of int or float representing the timestamps of spikes of a seq
    :param times_long_seq:
    :return:
    """

    for times_short in times_short_seq:
        not_in_any_long_times = True
        times_short = np.array(times_short)
        for times_long in times_long_seq:
            if (times_short[0] >= times_long[0]) and (times_short[-1] <= times_long[-1]):
                not_in_any_long_times = False
                break
            if len(np.intersect1d(times_short, np.array(times_long))):
                not_in_any_long_times = False
                break
        if not_in_any_long_times:
            return True
    return False


def removing_intersect_seq(set_seq_dict_result, min_len_seq, min_rep_nb, dict_by_len_seq):
    # set_seq_dict_result: each key represent a common seq (neurons tuple)
    # and each values represent a list of list of times
    # dict_by_len_seq: each key is an int representing the length of the seq contains
    # each value is a dict with as they key a tuple of int representing a neuron seq, and as value, a list
    # of list time instances
    keys_to_delete = set()
    for neurons_seq, times_seq_set in set_seq_dict_result.items():
        # in case it would have been already removed
        if neurons_seq not in set_seq_dict_result:
            continue
        if len(times_seq_set) < min_rep_nb:
            keys_to_delete.add(tuple(neurons_seq))
            continue
        if (len(times_seq_set) > 1) and (len(neurons_seq) > min_len_seq):
            # for each seq > at the min length, we take sub sequences starting from the last index
            # and look if it's present in the dictionnary with the same times
            for i in np.arange(len(neurons_seq) - min_len_seq) + 1:
                neurons_tuple_to_test = tuple(neurons_seq[0:-i])
                # print(f'neurons_tuple_to_test: {neurons_tuple_to_test}, ori: {neurons_seq}')
                if neurons_tuple_to_test in set_seq_dict_result:
                    orig_times = set()
                    # need to shorten the times to compare it later
                    for times in times_seq_set:
                        orig_times.add(times[0:-i])
                    times_to_check = set_seq_dict_result[neurons_tuple_to_test]
                    # print(f'times_seq_set: {orig_times}, times_to_check: {times_to_check}, '
                    #       f'eq: {times_to_check == orig_times}')
                    if times_to_check == orig_times:
                        # print(f'removing_intersect_seq neurons_seq {neurons_seq}, '
                        #       f'neurons_tuple_to_test {neurons_tuple_to_test}, times_seq_set: {orig_times}, '
                        #       f'times_to_check: {times_to_check}')
                        keys_to_delete.add(neurons_tuple_to_test)
    for key in keys_to_delete:
        if len(key) in dict_by_len_seq:
            dict_by_len_seq[len(key)].pop(key, None)
        set_seq_dict_result.pop(key, None)


def seq_intersecting_keys_of_dict(seq_dict, seq, is_totaly_include=False):
    is_found = False
    for s in seq_dict.keys():
        inter = np.intersect1d(np.asarray(seq), np.asarray(s))
        if is_totaly_include:
            if len(inter) == len(seq):
                is_found = True
                break
        else:
            if len(inter) > 0:
                is_found = True
                break
    return is_found


def no_other_seq_included(kept_seq_dict, seq_dict, seq_to_check, value):
    """

    :param kept_seq_dict:
    :param seq_dict:
    :param seq_to_check: seq_to_check not interestincg any kept_seq_dict keys, should have been tested before
    :param value:
    :return:
    """
    not_replaced = True
    seq_keys = list(seq_dict.keys())
    for seq_to_compare in seq_keys:
        if seq_to_compare not in seq_dict:
            continue

        if (len(seq_to_compare) <= len(seq_to_check)):
            continue

        # is_found = seq_intersecting_keys_of_dict(kept_seq_dict, seq=seq, is_totaly_include=True)
        inter = np.intersect1d(seq_ar, np.asarray(seq_to_compare))
        if len(inter) == len(seq):
            # it means the seq it totally include by seq_to_compare which is bigger
            # then if seq_to_compare is not intersecting any other seq we keep it instead of seq
            # otherwise we remove seq_to_compare
            is_found = seq_intersecting_keys_of_dict(kept_seq_dict, seq=seq_to_compare, is_totaly_include=False)
            if is_found:
                seq_dict.pop(seq_to_compare)
                continue
            # if not in sept_kept, then we need to see if any other seq is not including it and repeat it all
            other_value = seq_dict[seq_to_compare]
            seq_dict.pop(seq_to_compare)
            seq_res, value_res = no_other_seq_included(kept_seq_dict=kept_seq_dict, seq_dict=seq_dict,
                                                       seq_to_check=seq_to_compare, value=other_value)

            return seq_res, value_res

    return seq_to_check, value


def check_interesection_over_dict(kept_seq_dict, all_seq_dict, seq_to_check):
    """
    will verify that seq_to_check do not intersect with any seq kept_seq_dict, and then
    that no seq in all_seq_dict is includin seq_to_check, while not interesting  any seq kept_seq_dict
    :param kept_seq_dict:
    :param all_seq_dict:
    :param seq_to_check: tuple of int, should  be a key of all_seq_dict
    :return: True if the sequence if kept
    """
    value = None
    if seq_to_check in all_seq_dict:
        value = all_seq_dict[seq_to_check]
        all_seq_dict.pop(seq_to_check)
    else:
        return

    is_found = seq_intersecting_keys_of_dict(kept_seq_dict, seq=seq_to_check, is_totaly_include=False)
    if is_found:
        # means a seq kept already interesect the seq to check
        return

    # seq_ar = np.asarray(seq_to_check)

    seq, times = no_other_seq_included(kept_seq_dict=kept_seq_dict, seq_dict=all_seq_dict,
                                       seq_to_check=seq_to_check, value=value)
    if seq is not None:
        kept_seq_dict[seq] = times


def order_cells_from_seq_dict(seq_dict, non_ordered_neurons, param, debug_mode=False):
    """

    :param seq_dict: dict with key tuple of int representing neurons, and value a list of tuple, each tuple
    having the same length as the one as key and represents the times at which the neurons fire in that sequence
    :param non_ordered_neurons: list of neurons index (int) to be ordered
    :return:
    """
    if debug_mode:
        print(f"#### order_cells_from_seq_dict, len(non_ordered_neurons) {len(non_ordered_neurons)}, "
              f"non_ordered_neurons {non_ordered_neurons}")

    ordered_neurons = []
    non_ordered_neurons = np.asarray(non_ordered_neurons)
    min_len_for_new_seq = 2

    while (len(non_ordered_neurons) >= param.min_len_seq):
        if debug_mode:
            print(f"len(non_ordered_neurons) {len(non_ordered_neurons)}, "
                  f"param.min_len_seq {param.min_len_seq}")
        ordered_neurons_updated = False
        # key will be the number of neurons that are not in this interesection
        intersection_seq = SortedDict()
        # means the seq is entirely comprise in the not_ordered_neurons
        intersection_full_seq = SortedDict()
        no_intersection_found = True
        for neurons_tuple, times_tuples in seq_dict.items():
            inter = np.intersect1d(np.asarray(neurons_tuple), non_ordered_neurons)
            # only keeping the one for which len is >  to param.min_len_seq
            if len(inter) < min_len_for_new_seq:
                continue
            else:
                no_intersection_found = False
            # key represents the number of neurons in not_ordered_neurons not found in neurons_tuple
            # as the dict will be sorted by the keys
            key = len(non_ordered_neurons) - len(inter)
            if len(neurons_tuple) == len(inter):
                if key not in intersection_full_seq:
                    intersection_full_seq[key] = []
                intersection_full_seq[key].append(neurons_tuple)
            else:
                if key not in intersection_seq:
                    intersection_seq[key] = []
                intersection_seq[key].append(neurons_tuple)
        if no_intersection_found:
            break
        for key, value in intersection_full_seq.items():
            # value is a list of tuple representing cell indices
            # first keep the sequence with the most rep:
            best_seq = None
            best_nb_rep = 0
            for seq in value:
                times = seq_dict[seq]
                if len(times) > best_nb_rep:
                    best_nb_rep = len(times)
                    best_seq = seq
            if debug_mode:
                print(f"intersection_full_seq best_seq {best_seq}")
            ordered_neurons.extend(list(best_seq))
            non_ordered_neurons = np.setdiff1d(non_ordered_neurons, np.asarray(best_seq))
            ordered_neurons_updated = True
            break
        if not ordered_neurons_updated:
            # looking at seq that don't perfectly fetch non_ordered_neurons
            # print("all keys")
            # for key, list_seq in intersection_seq.items():
            #     print(f"key {key}")
            for key, list_seq in intersection_seq.items():
                if debug_mode:
                    print(f"// key {key}")
                # value is a list of tuple representing cell indices
                # first keep the sequence with the most rep:
                seq_to_add = None
                best_nb_rep = 0
                for seq in list_seq:
                    times = seq_dict[seq]
                    if len(times) > best_nb_rep:
                        best_nb_rep = len(times)
                        seq_to_add = seq
                inter = np.intersect1d(np.asarray(seq_to_add), non_ordered_neurons)
                new_seq = []
                mask_for_times = np.zeros(len(seq_to_add), dtype="bool")
                for i, neuron in enumerate(seq_to_add):
                    indices = np.where(non_ordered_neurons == neuron)[0]
                    # if that neuron exist in nn_ordered_neurons then we add it in the final list
                    # otherwise, we keep the mask to False to remove it later.
                    if len(indices) == 1:
                        index = indices[0]
                        mask_for_times[i] = True
                        new_seq.append(neuron)
                new_times_list = []
                for times_tuple in seq_dict[seq_to_add]:
                    # cchanging to np.array to use mask_for_times
                    tuple_as_array = np.asarray(times_tuple)
                    new_times_list.append(tuple(tuple_as_array[mask_for_times]))
                new_seq = tuple(new_seq)
                if debug_mode:
                    print(f"seq_to_add {seq_to_add}, new_seq {new_seq}, inter {inter}")
                if new_seq not in seq_dict:
                    seq_dict[new_seq] = new_times_list
                ordered_neurons.extend(list(new_seq))
                non_ordered_neurons = np.setdiff1d(non_ordered_neurons, inter)
                if debug_mode:
                    print(f"non_ordered_neurons updated: {non_ordered_neurons}")
                ordered_neurons_updated = True
                break

            if not ordered_neurons_updated:
                # it means nothing has changed, and won't changed
                break
    if debug_mode:
        print(f"final: ordered_neurons {ordered_neurons}, non_ordered_neurons {non_ordered_neurons}")
    ordered_neurons.extend(list(non_ordered_neurons))
    return ordered_neurons


def order_spike_nums_by_seq(spike_nums, param, sce_times_bool=None, debug_mode=True, reverse_order=False,
                            use_only_uniformity_method=False, just_keep_the_best=True,
                            use_loss_score_to_keep_the_best_from_tree=False,
                            keep_the_longest_seq=False, ms=None):
    """

    :param spike_nums:
    :param param: instance of MarkovParameters
    :param debug_mode:
    :param just_keep_the_best: if True, then don't feel the seq_dict
    :param just_keep_the_best: if True, doesn't return the dict with all sequences, just the best sequences
    :param use_loss_score_to_keep_the_best_from_tree: if True, then we keep sequences from the Tree sequence with the
    best lost_score, without looking at all sub-possibilities
    :return:
    """
    # ordered_spike_nums = spike_nums.copy()
    nb_neurons = len(spike_nums)
    if debug_mode:
        print("")
        print(f"nb_neurons {nb_neurons}")
    if nb_neurons == 0:
        return [], []

    # list_seq_dict

    list_seq_dict_uniform, dict_by_len_seq_uniform, max_seq_dict_uniform = \
        find_sequences(spike_nums=spike_nums, param=param,
                       try_uniformity_method=False, debug_mode=debug_mode,
                       sce_times_bool=sce_times_bool,
                       use_loss_score_to_keep_the_best=use_loss_score_to_keep_the_best_from_tree, ms=ms)

    if not use_only_uniformity_method:
        list_seq_dict, dict_by_len_seq, \
        max_seq_dict = find_sequences(spike_nums=spike_nums, param=param,
                                      debug_mode=debug_mode,
                                      sce_times_bool=sce_times_bool,
                                      use_loss_score_to_keep_the_best=use_loss_score_to_keep_the_best_from_tree,
                                      ms=ms)

    # list_seq_dict: list of length number of neurons, each elemeent represent a dictionnary containing
    # the sequences beginnning by this neuron

    # first we build a dictionnary from the dictionnary list
    seq_dict = dict()
    if not use_only_uniformity_method:
        for seq_dict_from_list in list_seq_dict:
            seq_dict.update(seq_dict_from_list)

    # adding sequences with algorithm taking into consideration un
    for seq_dict_from_list in list_seq_dict_uniform:
        seq_dict.update(seq_dict_from_list)

    if keep_the_longest_seq:
        max_len = np.max(list(dict_by_len_seq_uniform.keys()))
        if not use_only_uniformity_method:
            max_len = np.max((max_len, np.max(list(dict_by_len_seq.keys()))))
        best_seq = None
        best_seq_n_rep = 0
        if max_len in dict_by_len_seq_uniform:
            for seq, seq_times in dict_by_len_seq_uniform[max_len].items():
                if len(seq_times) > best_seq_n_rep:
                    best_seq = seq
                    best_seq_n_rep = len(seq_times)
        if (not use_only_uniformity_method) and (max_len in dict_by_len_seq):
            for seq, seq_times in dict_by_len_seq[max_len].items():
                if len(seq_times) > best_seq_n_rep:
                    best_seq = seq
                    best_seq_n_rep = len(seq_times)

        print(f"len(best_seq) {len(best_seq)}")
        new_order = np.zeros(nb_neurons, dtype="uint16")
        new_order[:len(best_seq)] = np.array(best_seq)

        non_ordered_neurons = np.setdiff1d(np.arange(nb_neurons),
                                           best_seq)
        if len(non_ordered_neurons) > 0:
            # will update seq_dict with new seq if necessary and give back an ordered_neurons list
            ordered_neurons = order_cells_from_seq_dict(seq_dict=seq_dict, non_ordered_neurons=non_ordered_neurons,
                                                        param=param, debug_mode=False)
            new_order[len(best_seq):] = ordered_neurons
            # new_order[len(seq):] = not_ordered_neurons
        return None, new_order, []

    # temporary code to test if taking the longest sequence is good option
    best_seq = None
    best_seq_cell = -1
    # keep seq for each cell
    all_best_seq = []
    best_loss_score = 1
    max_seq_dict_to_uses = [max_seq_dict_uniform]
    if not use_only_uniformity_method:
        max_seq_dict_to_uses.append(max_seq_dict)
    for max_seq_dict_to_use in max_seq_dict_to_uses:
        for k, seq in max_seq_dict_to_use.items():
            seq = np.array(seq)
            # print(f"{k} max_seq seq {seq}")
            # seq has not negative numbers
            new_order = np.zeros(nb_neurons, dtype="uint16")
            new_order[:len(seq)] = seq

            non_ordered_neurons = np.setdiff1d(np.arange(nb_neurons),
                                               seq)
            if len(non_ordered_neurons) > 0:
                # will update seq_dict with new seq if necessary and give back an ordered_neurons list
                ordered_neurons = order_cells_from_seq_dict(seq_dict=seq_dict, non_ordered_neurons=non_ordered_neurons,
                                                            param=param, debug_mode=False)
                new_order[len(seq):] = ordered_neurons
                # new_order[len(seq):] = not_ordered_neurons
            tmp_spike_nums = spike_nums[new_order, :]
            loss_score = loss_function_with_sliding_window(spike_nums=tmp_spike_nums[::-1, :],
                                                           time_inter_seq=param.time_inter_seq,
                                                           min_duration_intra_seq=param.min_duration_intra_seq)
            if debug_mode:
                print(f'loss_score neuron {k}, len {len(seq)}: {np.round(loss_score, 4)}')
            all_best_seq.append(np.array(new_order))
            if loss_score < best_loss_score:
                best_loss_score = loss_score
                best_seq = np.array(new_order)
                best_seq_cell = k

    if debug_mode:
        print(f'best loss_score neuron {best_seq_cell}: {np.round(best_loss_score, 4)}')

    # if debug_mode:
    #     print("####### all sequences #######")
    #     for key, value in seq_dict.items():
    #         print(f"seq: {key}, rep: {len(value)}")

    result_seq_dict = dict()
    if best_seq is not None and (not just_keep_the_best):

        selected_seq = get_seq_included_in(ref_seq=best_seq, seqs_to_test=list(seq_dict.keys()),
                                           min_len=param.min_len_seq, param=param)  # param.min_len_seq)
        result_seq_dict = dict()
        for seq in selected_seq:
            result_seq_dict[seq] = seq_dict[seq]

        other_version = False
        if other_version:
            # we want to filter list_seq_dict, keeping only sequences that are in best_seq and not interesting each other
            # starting by the seq that match the beginning of best_seq
            best_seq_len = len(best_seq)
            for i in np.arange(best_seq_len):
                seq_to_search = best_seq if i == 0 else best_seq[:-i]
                seq_to_search = tuple(seq_to_search)
                if seq_to_search in seq_dict:
                    result_seq_dict[seq_to_search] = seq_dict[seq_to_search]
                    seq_dict.pop(seq_to_search)
                    break
            # then going over each other seq
            # making sure it doesn't intersect with another seq already added and
            # that no seq not included doesn't include it

            # not to modify the dict while looping it
            seq_keys = list(seq_dict.keys())
            for seq in seq_keys:
                if seq in seq_dict:
                    times = seq_dict[seq]
                    # update dictionnaries
                    check_interesection_over_dict(kept_seq_dict=result_seq_dict,
                                                  all_seq_dict=seq_dict, seq_to_check=seq)

        if reverse_order:
            ordered_result_seq_dict = dict()
            # reversing all seq
            for key, values in result_seq_dict.items():
                new_key = key[::-1]
                new_values = []
                for v in values:
                    new_values.append(v[::-1])
                    ordered_result_seq_dict[new_key] = new_values
        else:
            ordered_result_seq_dict = result_seq_dict
        # then we have to do the same for new_list_seq_dict
    else:
        ordered_result_seq_dict = result_seq_dict
    if reverse_order:
        # we are reversing best seq, so the seq will appear from top to bottom
        best_seq = best_seq[::-1]

    if just_keep_the_best:
        return None, best_seq, all_best_seq
    # if test_new_version:
    return ordered_result_seq_dict, best_seq, all_best_seq
    # else:
    #     return list_seq_dict, best_seq


def get_seq_included_in(ref_seq, seqs_to_test, param, min_len=4):
    """
    Return all seq (list of tuple of int) from seqs_to_test that are in ref_seq (tuple of int), in the exact same order
    :param ref_seq:
    :param seq_to_test:
    :return:
    """

    result = []
    use_combinatory_without_errors_method = False
    if use_combinatory_without_errors_method:
        ref_combinations = []
        len_ref_seq = len(ref_seq)
        # first we compute all sequences that are combined in ref_seq
        for i in np.arange(len_ref_seq - min_len):
            for j in np.arange(i + min_len, len_ref_seq):
                ref_combinations.append(tuple(ref_seq[i:j]))

        for seq in seqs_to_test:
            # print(f"seq {seq} ref_combinations {ref_combinations}")
            if seq in ref_combinations:
                result.append(seq)
            # TODO: take in consideration an error rate
    else:
        for seq_to_test in seqs_to_test:
            nb_errors_so_far = 0
            last_index_found = -1
            i = 0
            for cell in seq_to_test:
                if nb_errors_so_far == (int(len(seq_to_test) * param.error_rate) + param.min_n_errors):
                    break
                i += 1
                cell_index = np.where(ref_seq == cell)[0]
                if len(cell_index) == 0:
                    if last_index_found > 0:
                        last_index_found += 1
                    nb_errors_so_far += 1
                    continue
                # otherwise cell_index should be equal to one as length only
                cell_index = cell_index[0]
                if last_index_found < 0:
                    last_index_found = cell_index
                if cell_index != (last_index_found + 1):
                    nb_errors_so_far += 1
                    if last_index_found > 0:
                        last_index_found += 1
                    continue
            if i == len(seq_to_test):
                # then we keep it
                result.append(seq_to_test)
    return result


def find_significant_patterns(spike_nums, param, activity_threshold, sliding_window_duration,
                              data_id, n_surrogate=2, extra_file_name="", debug_mode=False, without_raw_plot=True,
                              sce_times_bool=None,
                              use_ordered_spike_nums_for_surrogate=False,
                              use_only_uniformity_method=False,
                              labels=None,
                              use_loss_score_to_keep_the_best_from_tree=
                              False, spike_shape="|",
                              spike_shape_size=10,
                              jitter_links_range=5,
                              keep_the_longest_seq=False, ms=None):
    if labels is None:
        labels = np.arange(len(spike_nums))

    if not without_raw_plot:
        plot_spikes_raster(spike_nums=spike_nums, param=param,
                           spike_train_format=False,
                           title=f"raster plot {data_id}",
                           file_name=f"raw_spike_nums_{data_id}{extra_file_name}",
                           y_ticks_labels=labels,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=activity_threshold,
                           # 500 ms window
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape=spike_shape,
                           spike_shape_size=spike_shape_size,
                           save_formats="pdf")
    # continue

    # 2128885
    loss_score = loss_function_with_sliding_window(spike_nums=spike_nums,
                                                   time_inter_seq=param.time_inter_seq,
                                                   spike_train_mode=False,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   debug_mode=debug_mode)

    print(f'raw loss_score: {np.round(loss_score, 4)}')

    # spike_struct.spike_data = trains_module.from_spike_trains_to_spike_nums(spike_struct.spike_data)
    # [:, :8000]
    best_seq_real_data, seq_dict_real_data = sort_it_and_plot_it(spike_nums=spike_nums, param=param,
                                                                 sliding_window_duration=sliding_window_duration,
                                                                 activity_threshold=activity_threshold,
                                                                 title_option=f"{data_id}{extra_file_name}",
                                                                 spike_train_format=False,
                                                                 debug_mode=debug_mode,
                                                                 labels=labels,
                                                                 sce_times_bool=sce_times_bool,
                                                                 use_only_uniformity_method=use_only_uniformity_method,
                                                                 use_loss_score_to_keep_the_best_from_tree=
                                                                 use_loss_score_to_keep_the_best_from_tree,
                                                                 spike_shape=spike_shape,
                                                                 spike_shape_size=spike_shape_size,
                                                                 keep_the_longest_seq=keep_the_longest_seq,
                                                                 ms=ms)

    nb_cells = len(spike_nums)

    print("#### REAL DATA ####")
    if debug_mode:
        print(f"best_seq {best_seq_real_data}")

    ordered_labels_real_data = []
    for old_cell_index in best_seq_real_data:
        ordered_labels_real_data.append(labels[old_cell_index])

    real_data_result_for_stat = SortedDict()
    neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")
    if seq_dict_real_data is not None:
        for key, value in seq_dict_real_data.items():
            # print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
            if len(key) not in real_data_result_for_stat:
                real_data_result_for_stat[len(key)] = dict()
                real_data_result_for_stat[len(key)]["rep"] = []
                real_data_result_for_stat[len(key)]["duration"] = []
            real_data_result_for_stat[len(key)]["rep"].append(len(value))
            list_of_durations = []
            # keeping the duration of each repetition
            for time_stamps in value:
                list_of_durations.append(time_stamps[-1] - time_stamps[0])
            real_data_result_for_stat[len(key)]["duration"].append(list_of_durations)
            for cell in key:
                if neurons_sorted_real_data[cell] == 0:
                    neurons_sorted_real_data[cell] = 1

    n_times = len(spike_nums[0, :])

    print("#### SURROGATE DATA ####")
    # n_surrogate = 2
    surrogate_data_result_for_stat = SortedDict()
    neurons_sorted_surrogate_data = np.zeros(nb_cells, dtype="uint16")
    nb_seq_by_len_for_each_surrogate = np.zeros((nb_cells + 1, n_surrogate), dtype="uint16")
    for surrogate_number in np.arange(n_surrogate):
        print(f"#### SURROGATE n° {surrogate_number} ####")
        if use_ordered_spike_nums_for_surrogate:
            #
            do_roll_option = True
            if do_roll_option:
                # using ordered spike_nums that we will surrogate
                copy_spike_nums = np.copy(spike_nums[best_seq_real_data, :])
                for n, neuron_spikes in enumerate(copy_spike_nums):
                    # roll the data to a random displace number
                    copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            else:
                # we shuffle cells instead
                best_seq_copy = np.copy(best_seq_real_data)
                np.random.shuffle(best_seq_copy)
                copy_spike_nums = np.copy(spike_nums[best_seq_copy, :])
            # [:, :8000]
            seq_dict_surrogate = find_sequences_in_ordered_spike_nums(spike_nums=copy_spike_nums,
                                                                      param=param)
        else:
            copy_spike_nums = np.copy(spike_nums)
            for n, neuron_spikes in enumerate(copy_spike_nums):
                # roll the data to a random displace number
                copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
            tmp_spike_nums = copy_spike_nums
            save_plots = False if (surrogate_number > 0) else True
            best_seq_surrogate, seq_dict_surrogate = \
                sort_it_and_plot_it(spike_nums=tmp_spike_nums, param=param,
                                    sliding_window_duration=sliding_window_duration,
                                    activity_threshold=activity_threshold,
                                    title_option=f"surrogate_{surrogate_number}_"
                                    f"{data_id}{extra_file_name}",
                                    spike_train_format=False,
                                    debug_mode=False,
                                    use_only_uniformity_method=use_only_uniformity_method,
                                    use_loss_score_to_keep_the_best_from_tree=
                                    use_loss_score_to_keep_the_best_from_tree,
                                    save_plots=save_plots,
                                    spike_shape=spike_shape,
                                    spike_shape_size=spike_shape_size,
                                    )

        # print(f"best_seq {best_seq_surrogate}")

        mask = np.zeros(nb_cells, dtype="bool")
        if seq_dict_surrogate is not None:
            for key, value in seq_dict_surrogate.items():
                # print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
                # counting the number of a given length for each surrogate
                nb_seq_by_len_for_each_surrogate[len(key), surrogate_number] += 1
                if len(key) not in surrogate_data_result_for_stat:
                    surrogate_data_result_for_stat[len(key)] = dict()
                    surrogate_data_result_for_stat[len(key)]["rep"] = []
                    surrogate_data_result_for_stat[len(key)]["duration"] = []
                surrogate_data_result_for_stat[len(key)]["rep"].append(len(value))
                # keeping the duration of each repetition
                list_of_durations = []
                for time_stamps in value:
                    list_of_durations.append(time_stamps[-1] - time_stamps[0])
                surrogate_data_result_for_stat[len(key)]["duration"].append(list_of_durations)
                for cell in key:
                    mask[cell] = True

            neurons_sorted_surrogate_data[mask] += 1
    # min_time, max_time = trains_module.get_range_train_list(spike_nums)
    # surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
    #                                               min_value=min_time, max_value=max_time)
    print("")
    print("")

    significant_threshold_by_seq_len_and_rep = dict()

    for key in surrogate_data_result_for_stat.keys():
        significant_threshold_by_seq_len_and_rep[key] = np.percentile(surrogate_data_result_for_stat[key]["rep"], 95)

    # filtering seq to keep only the significant one
    significant_seq_dict = dict()
    # for each sequence gives its significance level
    # 1 if significant due the number of seq of this length in the dataset
    # 2 if significant due the number of repetition of this sequence in comparison of same length seq from surrogate
    # 3 if significant for the both previous reason
    # 4 if significant because no seq of this length found in the surrogate
    significant_category_dict = dict()
    for cells, times in seq_dict_real_data.items():
        category = 0

        if len(cells) not in significant_threshold_by_seq_len_and_rep:
            category = 4
        else:
            threshold_len = np.percentile(nb_seq_by_len_for_each_surrogate[len(cells), :], 95)
            if len(real_data_result_for_stat[len(cells)]["rep"]) > threshold_len:
                category += 1

            if len(times) > significant_threshold_by_seq_len_and_rep[len(cells)]:
                category += 2
        if category > 0:
            significant_seq_dict[cells] = times
            significant_category_dict[cells] = category

    real_data_significant_result_for_stat = SortedDict()
    neurons_sorted_real_data = np.zeros(nb_cells, dtype="uint16")

    significant_category_dict_by_len = dict()
    for key, value in significant_seq_dict.items():
        # print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
        if len(key) not in real_data_significant_result_for_stat:
            real_data_significant_result_for_stat[len(key)] = []
            significant_category_dict_by_len[len(key)] = []
        real_data_significant_result_for_stat[len(key)].append(len(value))
        significant_category_dict_by_len[len(key)].append(significant_category_dict[key])
        for cell in key:
            if neurons_sorted_real_data[cell] == 0:
                neurons_sorted_real_data[cell] = 1

    give_me_stat_on_sorting_seq_results(results_dict=real_data_result_for_stat,
                                        significant_results_dict=real_data_significant_result_for_stat,
                                        significant_category_dict_by_len=significant_category_dict_by_len,
                                        significant_seq_dict=significant_seq_dict,
                                        significant_category_dict=significant_category_dict,
                                        labels=ordered_labels_real_data,
                                        best_cells_order=best_seq_real_data,
                                        neurons_sorted=neurons_sorted_real_data,
                                        title=f"%%%% DATA SET STAT {data_id} %%%%%", param=param,
                                        results_dict_surrogate=surrogate_data_result_for_stat,
                                        neurons_sorted_surrogate=neurons_sorted_surrogate_data,
                                        extra_file_name=data_id + extra_file_name,
                                        n_surrogate=n_surrogate,
                                        use_sce_times_for_pattern_search=(sce_times_bool is not None),
                                        use_only_uniformity_method=use_only_uniformity_method,
                                        use_loss_score_to_keep_the_best_from_tree=
                                        use_loss_score_to_keep_the_best_from_tree,
                                        use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate,
                                        keep_the_longest_seq=keep_the_longest_seq
                                        )

    # seq_dict_real_data_backup = dict()
    # seq_dict_real_data_backup.update(seq_dict_real_data)

    title_option = f"{extra_file_name}_significant_seq"
    colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                           "magenta"]
    spike_nums_ordered = spike_nums[best_seq_real_data, :]

    plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                       title=f"raster plot ordered {data_id} {extra_file_name} {title_option}",
                       spike_train_format=False,
                       file_name=f"{data_id}_{extra_file_name}spike_nums_ordered_seq_{title_option}",
                       y_ticks_labels=ordered_labels_real_data,
                       save_raster=True,
                       show_raster=False,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=activity_threshold,
                       save_formats="pdf",
                       seq_times_to_color_dict=significant_seq_dict,
                       link_seq_categories=significant_category_dict,
                       link_seq_color=colors_for_seq_list,
                       link_seq_line_width=0.6,
                       link_seq_alpha=0.9,
                       jitter_links_range=jitter_links_range,
                       min_len_links_seq=3,
                       spike_shape=spike_shape,
                       spike_shape_size=spike_shape_size)


def sort_it_and_plot_it(spike_nums, param,
                        sliding_window_duration, activity_threshold, title_option="",
                        sce_times_bool=None,
                        spike_train_format=False,
                        debug_mode=False,
                        plot_all_best_seq_by_cell=False,
                        use_only_uniformity_method=False,
                        use_loss_score_to_keep_the_best_from_tree=
                        False,
                        labels=None,
                        save_plots=True, spike_shape="|",
                        spike_shape_size=10, keep_the_longest_seq=False, ms=None
                        ):
    if spike_train_format:
        return
    # if sce_times_bool is not None, then we don't take in consideration SCE_time to do the pair-wise correlation
    result = order_spike_nums_by_seq(spike_nums,
                                     param, sce_times_bool=sce_times_bool,
                                     debug_mode=debug_mode,
                                     use_only_uniformity_method=use_only_uniformity_method,
                                     use_loss_score_to_keep_the_best_from_tree=
                                     use_loss_score_to_keep_the_best_from_tree,
                                     keep_the_longest_seq=keep_the_longest_seq, ms=ms)
    seq_dict_tmp, best_seq, all_best_seq = result

    if labels is None:
        labels = np.arange(len(spike_nums))
    ordered_labels = []
    for old_cell_index in best_seq:
        ordered_labels.append(labels[old_cell_index])

    if plot_all_best_seq_by_cell:
        for cell, each_best_seq in enumerate(all_best_seq):
            spike_nums_ordered = np.copy(spike_nums[each_best_seq, :])

            loss_score = loss_function_with_sliding_window(spike_nums=spike_nums_ordered[::-1, :],
                                                           time_inter_seq=param.time_inter_seq,
                                                           min_duration_intra_seq=param.min_duration_intra_seq,
                                                           spike_train_mode=False,
                                                           debug_mode=True
                                                           )
            if debug_mode:
                print(f'Cell {cell} loss_score ordered: {np.round(loss_score, 4)}')
            # saving the ordered spike_nums
            # micro_wires_ordered = micro_wires[best_seq]
            # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
            #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

            plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                               title=f"cell {cell} raster plot ordered {title_option}",
                               spike_train_format=False,
                               file_name=f"cell_{cell}_spike_nums_ordered_{title_option}",
                               y_ticks_labels=ordered_labels,
                               save_raster=True,
                               show_raster=False,
                               sliding_window_duration=sliding_window_duration,
                               show_sum_spikes_as_percentage=True,
                               plot_with_amplitude=False,
                               spike_shape=spike_shape,
                               spike_shape_size=spike_shape_size,
                               activity_threshold=activity_threshold,
                               save_formats="pdf")
    else:
        spike_nums_ordered = np.copy(spike_nums[best_seq, :])

        if save_plots:
            plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                               title=f"raster plot ordered {title_option}",
                               spike_train_format=False,
                               file_name=f"spike_nums_ordered_{title_option}",
                               y_ticks_labels=ordered_labels,
                               save_raster=True,
                               show_raster=False,
                               sliding_window_duration=sliding_window_duration,
                               show_sum_spikes_as_percentage=True,
                               plot_with_amplitude=False,
                               spike_shape=spike_shape,
                               spike_shape_size=spike_shape_size,
                               activity_threshold=activity_threshold,
                               save_formats="pdf")

    spike_nums_ordered = np.copy(spike_nums[best_seq, :])
    if debug_mode:
        print(f"starting finding sequences in orderered spike nums")
    seq_dict = find_sequences_in_ordered_spike_nums(spike_nums=spike_nums_ordered, param=param)
    if debug_mode:
        print(f"Sequences in orderered spike nums found")
    if not save_plots:
        return best_seq, seq_dict
    # if debug_mode:
    #     print(f"best_seq {best_seq}")
    # if seq_dict_tmp is not None:
    #     if debug_mode:
    #         for key, value in seq_dict_tmp.items():
    #             print(f"seq: {key}, rep: {len(value)}")
    #
    #     best_seq_mapping_index = dict()
    #     for i, cell in enumerate(best_seq):
    #         best_seq_mapping_index[cell] = i
    #     # we need to replace the index by the corresponding one in best_seq
    #     seq_dict = dict()
    #     for key, value in seq_dict_tmp.items():
    #         new_key = []
    #         for cell in key:
    #             new_key.append(best_seq_mapping_index[cell])
    #         # checking if the list of cell is in the same order in best_seq
    #         # if the diff is only composed of one, this means all indices are following each other
    #         in_order = len(np.where(np.diff(new_key) != 1)[0]) == 0
    #         if in_order:
    #             print(f"in_order {new_key}")
    #             seq_dict[tuple(new_key)] = value
    #
    #     seq_colors = dict()
    #     len_seq = len(seq_dict)
    #     if debug_mode:
    #         print(f"nb seq to colors: {len_seq}")
    #     for index, key in enumerate(seq_dict.keys()):
    #         seq_colors[key] = cm.nipy_spectral(float(index + 1) / (len_seq + 1))
    #         if debug_mode:
    #             print(f"color {seq_colors[key]}, len(seq) {len(key)}")
    # else:
    #     seq_dict = None
    #     seq_colors = None
    # ordered_spike_nums = ordered_spike_data
    # spike_struct.ordered_spike_data = \
    #     trains_module.from_spike_nums_to_spike_trains(spike_struct.ordered_spike_data)

    loss_score = loss_function_with_sliding_window(spike_nums=spike_nums_ordered,
                                                   time_inter_seq=param.time_inter_seq,
                                                   min_duration_intra_seq=param.min_duration_intra_seq,
                                                   spike_train_mode=False,
                                                   debug_mode=True
                                                   )
    if debug_mode:
        print(f'total loss_score ordered: {np.round(loss_score, 4)}')
    # saving the ordered spike_nums
    # micro_wires_ordered = micro_wires[best_seq]
    # np.savez(f'{param.path_results}/{channels_selection}_spike_nums_ordered_{patient_id}.npz',
    #          spike_nums_ordered=spike_nums_ordered, micro_wires_ordered=micro_wires_ordered)

    colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                           "magenta"]
    plot_spikes_raster(spike_nums=spike_nums_ordered, param=param,
                       title=f"raster plot ordered {title_option}",
                       spike_train_format=False,
                       file_name=f"spike_nums_ordered_seq_{title_option}",
                       y_ticks_labels=ordered_labels,
                       save_raster=True,
                       show_raster=False,
                       spike_shape=spike_shape,
                       spike_shape_size=spike_shape_size,
                       sliding_window_duration=sliding_window_duration,
                       show_sum_spikes_as_percentage=True,
                       plot_with_amplitude=False,
                       activity_threshold=activity_threshold,
                       save_formats="pdf",
                       seq_times_to_color_dict=seq_dict,
                       link_seq_color=colors_for_seq_list,
                       link_seq_line_width=0.8,
                       link_seq_alpha=0.9,
                       jitter_links_range=5,
                       min_len_links_seq=3)
    # seq_colors=seq_colors)

    return best_seq, seq_dict
