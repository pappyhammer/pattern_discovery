import numpy as np
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.param as p_disc_tools_param
import pattern_discovery.tools.misc as p_disc_tools_misc
from pattern_discovery.structures.seq_tree_v1 import Tree
from sortedcontainers import SortedList, SortedDict
from pattern_discovery.display.raster import plot_spikes_raster


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


def build_mle_transition_dict(spike_nums, param, sce_times_bool=None,
                              try_uniformity_method=False, debug_mode=False):
    """
    Maximum Likelihood estimation,
    don't take into account the fact that if a neuron A fire after a neuron B ,
    then it decreases the probability than B fires after A
    :param spike_nums:
    :param param:
    :return:
    """
    nb_neurons = len(spike_nums)
    n_times = len(spike_nums[0, :])
    transition_dict = np.zeros((nb_neurons, nb_neurons))
    # results obtain from uniform distribution
    uniform_transition_dict = np.zeros((nb_neurons, nb_neurons))
    # so the neuron with the lower spike rates gets the biggest weight in terms of probability
    if param.spike_rate_weight and (not try_uniformity_method):
        spike_rates = 1 - p_disc_tools_misc.get_spike_rates(spike_nums)
    else:
        spike_rates = np.ones(nb_neurons)

    # we must first introduce a frequency correction
    # using a uniform_spike_nums with uniformly distributed spike trains having the same spike frequency
    uniform_spike_nums = np.zeros((nb_neurons, n_times), dtype="uint8")
    for n, neuron in enumerate(spike_nums):
        times = np.where(neuron)[0]
        nb_times = len(times)
        if nb_times > 0:
            delta_t = (n_times + 1) // nb_times
            new_spike_times = np.arange(0, nb_times, delta_t)
            uniform_spike_nums[n, new_spike_times] = 1

    # a first turn to put probabilities up from neurons B that spikes after neuron A
    for n, neuron in enumerate(spike_nums):
        # will count how many spikes of each neuron are following the spike of
        # tmp_count = np.zeros(nb_neurons)
        for t in np.where(neuron)[0]:
            original_t = t
            t = np.max((0, t + param.min_duration_intra_seq))
            t_max = np.min((t + param.time_inter_seq, len(spike_nums[0]) - 1))
            times_to_check = np.arange(t, t_max)
            # TODO: do something similar to Robin, to detect events, with percentile
            if sce_times_bool is not None:
                # if > 0, means there is a SCE during that interval, and we don't count it
                # if np.sum(sce_times_bool[t:t_max]) > 0:
                #         continue

                # another option is to remove the times of the SCE from the search:
                times_to_check = times_to_check[sce_times_bool[t:t_max]]
                if len(times_to_check) == 0:
                    continue

            actual_neurons_spikes = spike_nums[n, :] > 0
            spike_nums[n, actual_neurons_spikes] = 0

            pos = np.where(spike_nums[:, times_to_check])[0]
            # pos = np.unique(pos)
            for p in pos:
                transition_dict[n, p] = transition_dict[n, p] + spike_rates[p]
                if param.no_reverse_seq:
                    # see to put transition to 0 ??
                    transition_dict[p, n] = transition_dict[p, n] - spike_rates[p]
            # transition_dict[n, pos] = transition_dict[n, pos] + 1

            spike_nums[n, actual_neurons_spikes] = 1

        if try_uniformity_method:
            for t in np.where(uniform_spike_nums[n, :])[0]:
                original_t = t
                t = np.max((0, t + param.min_duration_intra_seq))
                t_max = np.min((t + param.time_inter_seq, len(spike_nums[0]) - 1))

                times_to_check = np.arange(t, t_max)
                # TODO: do something similar to Robin, to detect events, with percentile
                if sce_times_bool is not None:
                    # if > 0, means there is a SCE during that interval, and we don't count it
                    # if np.sum(sce_times_bool[t:t_max]) > 0:
                    #         continue

                    # another option is to remove the times of the SCE from the search:
                    times_to_check = times_to_check[sce_times_bool[t:t_max]]
                    if len(times_to_check) == 0:
                        continue

                actual_neurons_spikes = spike_nums[n, :] > 0
                spike_nums[n, actual_neurons_spikes] = 0

                pos = np.where(spike_nums[:, times_to_check])[0]
                # pos = np.unique(pos)
                for p in pos:
                    uniform_transition_dict[n, p] = uniform_transition_dict[n, p] + spike_rates[p]
                    if param.no_reverse_seq:
                        # see to put transition to 0 ??
                        uniform_transition_dict[p, n] = uniform_transition_dict[p, n] - spike_rates[p]
                # transition_dict[n, pos] = transition_dict[n, pos] + 1
                spike_nums[n, actual_neurons_spikes] = 1
            uniform_transition_dict[n, n] = 0
        transition_dict[n, n] = 0

    # we divide for each neuron by the sum of the probabilities
    for n, neuron in enumerate(spike_nums):
        # print(f'n {n}, len(np.where(transition_dict[n, :] < 0)[0]) {len(np.where(transition_dict[n, :] < 0)[0])}')
        # all negatives values should be put to zero
        transition_dict[n, np.where(transition_dict[n, :] < 0)[0]] = 0
        if try_uniformity_method:
            uniform_transition_dict[n, np.where(uniform_transition_dict[n, :] < 0)[0]] = 0
            other_neurons = np.where(uniform_transition_dict[n, :])[0]
            for o_n in other_neurons:
                # if transition_dict[n, o_n] > 0:
                #     print(f"n {n}, t {o_n}, transition_dict[n, o_n] {transition_dict[n, o_n]}"
                #           f", uniform_transition_dict[n, o_n] {uniform_transition_dict[n, o_n]}")
                transition_dict[n, o_n] = transition_dict[n, o_n] / uniform_transition_dict[n, o_n]
        if np.sum(transition_dict[n, :]) > 0:
            transition_dict[n, :] = transition_dict[n, :] / np.sum(transition_dict[n, :])
        else:
            print(f"np.sum(transition_dict[n, :]) <= 0: {np.sum(transition_dict[n, :])}")

    print_transit_dict = False
    if print_transit_dict:
        for n, neuron in enumerate(spike_nums):
            print(f'transition dict, n {n}, sum: {np.sum(transition_dict[n, :])}')
            print(f'transition dict, n {n}, max: {np.max(transition_dict[n, :])}')
            print(f'transition dict, n {n}, nb max: '
                  f'{np.where(transition_dict[n, :] == np.max(transition_dict[n, :]))[0]}')
    if debug_mode:
        print(f'median transition: {np.median(transition_dict)}')
        print(f'mean transition: {np.mean(transition_dict)}')
        print(f'std transition: {np.std(transition_dict)}')
        print(f'min transition: {np.min(transition_dict)}')
        print(f'max transition: {np.max(transition_dict)}')
    return transition_dict


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
                                        percentile_threshold=95):
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
                str_to_write += f"# Surrogate (nb seq: {np.round((len(nb_rep_seq_surrogate)/n_surrogate), 4)}), " \
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

    file_name = f'{param.path_results}/significant_sorting_results_with_timestamps{extra_file_name}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write("best_order:")
        for cell_id, cell in enumerate(best_cells_order):
            file.write(f"{cell}")
            if cell_id < (len(best_cells_order) - 1):
                file.write(" ")
        file.write("\n")
        for cells, value in significant_seq_dict.items():
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


def find_sequences(spike_nums, param, sce_times_bool=None, try_uniformity_method=False, debug_mode=False,
                   use_loss_score_to_keep_the_best=False):
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
    dict_by_len_seq = dict()

    # print(f'nb spikes neuron 1: {len(np.where(spike_nums[1,:])[0])}')

    # spike_nums_backup = spike_nums
    spike_nums = np.copy(spike_nums)
    # spike_nums = spike_nums[:, :2000]
    nb_neurons = len(spike_nums)

    transition_dict = build_mle_transition_dict(spike_nums=spike_nums, param=param,
                                                try_uniformity_method=try_uniformity_method,
                                                debug_mode=debug_mode,
                                                sce_times_bool=sce_times_bool)
    # print(f"transition_dict {transition_dict}")
    # len of nb_neurons, each element is a dictionary with each key represent a common seq (neurons tuple, first neurons
    # being the index of the list)
    # and each values represent a list of list of times
    list_dict_result = []

    # key: neuron as integer, value: list of neurons being the longest probable sequence
    max_seq_dict = dict()
    # Start to look for real sequences in spike_nums
    for y, neuron in enumerate(spike_nums):
        tree = bfs(trans_dict=transition_dict, neuron_to_start=y, param=param)
        # from tree to list of list
        sequences = tree.get_seq_lists()
        if debug_mode:
            print(f'Nb probabilities seq for neuron {y}: {len(sequences)}')
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
            max_seq_dict[y] = best_seq

        for seq in sequences:
            # print(f"Neuron {y}, tree seq: {seq}")
            # look for each spike of this neuron, if a seq is found on the following spikes of other neurons
            for t in np.where(neuron)[0]:
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
                                max_seq_dict[y] = seq
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
            print(f'Len max for neuron {y} prob seq: {max_len_prob_seq}')
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

    # max_rep_non_prob is the maximum rep of a sequence found in spike_nums
    return list_dict_result, dict_by_len_seq, max_seq_dict


def find_sequences_in_ordered_spike_nums(spike_nums, param):
    """
    Find sequence in spike_nums starting from cell 0, with respect of param (such as len_seq, rep_seq etc..)
    :param spike_nums:
    :param param:
    :return:
    """
    seq_dict = dict()
    n_cells = len(spike_nums)
    n_times = len(spike_nums[0, :])

    for cell_id, cell_times in enumerate(spike_nums):
        if (n_cells - 1 - cell_id) < param.min_len_seq:
            break
        current_seq_dict = dict()
        cell_spikes = np.where(cell_times)[0]
        for spike_id, spike_time in enumerate(cell_spikes):
            # to avoid the same spike if using rasterdur
            if spike_id > 0:
                if spike_time == (cell_spikes[spike_id - 1] + 1):
                    continue
            last_spike_time = spike_time
            nb_errors = 0
            errors_index = []
            index_seq = 0
            current_seq_cells = [cell_id]
            current_seq_times = [spike_time]
            time_inter_seq = param.time_inter_seq
            for next_cell_id in np.arange(cell_id + 1, n_cells):
                min_time = np.max((0, last_spike_time + param.min_duration_intra_seq))
                max_time = np.min((n_times, last_spike_time + time_inter_seq))

                spikes_next_cell = np.where(spike_nums[next_cell_id, min_time:max_time])[0]
                if len(spikes_next_cell) > 0:
                    current_seq_times.append(spikes_next_cell[0] + min_time)
                    last_spike_time = spikes_next_cell[0] + min_time
                    current_seq_cells.append(next_cell_id)
                    time_inter_seq = param.time_inter_seq
                else:
                    if nb_errors < (int(len(current_seq_cells) * param.error_rate) + param.min_n_errors):
                        nb_errors += 1
                        errors_index.append(index_seq)
                        current_seq_cells.append(next_cell_id)
                        # put a fake time, where no spike exist for this cell
                        current_seq_times.append(min_time)
                        time_inter_seq += param.time_inter_seq
                    else:
                        break

                index_seq += 1

            # first if errors have been added at the end, we remove them
            while len(errors_index) > 0:
                if errors_index[-1] == (len(current_seq_cells) - 1):
                    current_seq_cells = current_seq_cells[:-1]
                    current_seq_times = current_seq_times[:-1]
                    errors_index = errors_index[:-1]
                else:
                    break

            # if too many errors comparing to the length of the seq, we don't keep it
            if len(errors_index) > (int(len(current_seq_cells) * param.error_rate) + param.min_n_errors):
                # print(f"len(errors_index) > int(len(current_seq_cells) * param.error_rate) "
                #       f"len(errors_index) {len(errors_index)}, "
                #       f"int(len(current_seq_cells) * param.error_rate) "
                #       f"{int(len(current_seq_cells) * param.error_rate)}")
                continue

            # then we check if the seq has the min length
            if len(current_seq_cells) > param.min_len_seq:
                # if we haven't reach the max errors, we check if by adding errors before we could add it
                # to a seq already existing
                not_added = True
                current_seq_cells_backup = current_seq_cells[:]
                current_seq_times_backup = current_seq_times[:]
                nb_errors_to_add = int(len(current_seq_cells) * param.error_rate) + param.min_n_errors - \
                                   len(errors_index)
                while not_added and (nb_errors_to_add >= 0):
                    first_cell = current_seq_cells_backup[0]
                    if (nb_errors_to_add > 0) and ((first_cell - nb_errors_to_add) >= 0):
                        first_cell_time = current_seq_times_backup[0]
                        current_seq_cells = list(np.arange(first_cell - nb_errors_to_add, first_cell)) + \
                                            current_seq_cells_backup
                        current_seq_times = ([first_cell_time] * nb_errors_to_add) + current_seq_times_backup
                    else:
                        current_seq_cells = current_seq_cells_backup
                        current_seq_times = current_seq_times_backup
                    nb_errors_to_add -= 1

                    tuple_seq = tuple(current_seq_cells)

                    # we add errors before only in order to see if it match sequences already added with previous
                    # cells
                    if (tuple_seq not in current_seq_dict) and nb_errors_to_add >= 0:
                        continue

                    if tuple_seq not in current_seq_dict:
                        current_seq_dict[tuple_seq] = []
                        current_seq_dict[tuple_seq].append(current_seq_times)
                        not_added = False
                    else:
                        # first we check that the times of the new seq are no intersect with other one
                        ok_to_add_it = True
                        for times_seq_already_in in current_seq_dict[tuple_seq]:
                            for time_id, time_value in enumerate(times_seq_already_in):
                                # we can't use intersect of setdiff
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
                            not_added = False

        # we need to filter the dict to remove seq that don't repeat enough
        seq_to_remove = []
        seq_to_remove_from_valid_seq = []
        for key, value in current_seq_dict.items():
            if len(value) < param.min_rep_nb:
                seq_to_remove.append(key)
                continue
            # check if there is no intersection with seq already in seq_dict
            for valid_seq, valid_times in seq_dict.items():
                if len(key) <= len(valid_seq):
                    unique_cells = np.setdiff1d(key, valid_seq)
                    if len(unique_cells) == 0:
                        # we want to see if the shorter seq is always at the same time of the longer one
                        # if not, then we keep it
                        if not is_seq_independant(times_short_seq=value, times_long_seq=valid_times):
                            seq_to_remove.append(key)
                else:
                    unique_cells = np.setdiff1d(valid_seq, key)
                    if len(unique_cells) == 0:
                        if not is_seq_independant(times_short_seq=valid_times, times_long_seq=value):
                            seq_to_remove_from_valid_seq.append(valid_seq)

        for key in seq_to_remove:
            if key in current_seq_dict:
                del current_seq_dict[key]

        for key in seq_to_remove_from_valid_seq:
            if key in seq_dict:
                del seq_dict[key]

        seq_dict.update(current_seq_dict)

    return seq_dict


def is_seq_independant(times_short_seq, times_long_seq):
    """

    :param times_short_seq: list of list of int or float representing the timestamps of spikes of a seq
    :param times_long_seq:
    :return:
    """

    for times_short in times_short_seq:
        not_in_any_long_times = True
        for times_long in times_long_seq:
            if (times_short[0] >= times_long[0]) and (times_short[-1] <= times_long[-1]):
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
                            use_loss_score_to_keep_the_best_from_tree=False):
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
                       try_uniformity_method=True, debug_mode=debug_mode,
                       sce_times_bool=sce_times_bool,
                       use_loss_score_to_keep_the_best=use_loss_score_to_keep_the_best_from_tree)

    if not use_only_uniformity_method:
        list_seq_dict, dict_by_len_seq, \
        max_seq_dict = find_sequences(spike_nums=spike_nums, param=param,
                                      debug_mode=debug_mode,
                                      sce_times_bool=sce_times_bool,
                                      use_loss_score_to_keep_the_best=use_loss_score_to_keep_the_best_from_tree)

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
                              jitter_links_range=5):
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
                                                                 spike_shape_size=spike_shape_size, )

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
                                        use_ordered_spike_nums_for_surrogate=use_ordered_spike_nums_for_surrogate
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
                        spike_shape_size=10
                        ):
    if spike_train_format:
        return
    # if sce_times_bool is not None, then we don't take in consideration SCE_time to do the pair-wise correlation
    result = order_spike_nums_by_seq(spike_nums,
                                     param, sce_times_bool=sce_times_bool,
                                     debug_mode=debug_mode,
                                     use_only_uniformity_method=use_only_uniformity_method,
                                     use_loss_score_to_keep_the_best_from_tree=
                                     use_loss_score_to_keep_the_best_from_tree)
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
