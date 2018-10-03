import numpy as np
from pattern_discovery.tools.loss_function import loss_function_with_sliding_window
import pattern_discovery.tools.param as p_disc_tools_param
import pattern_discovery.tools.misc as p_disc_tools_misc
from pattern_discovery.structures.seq_tree_v1 import Tree
from sortedcontainers import SortedList, SortedDict


class MarkovParameters(p_disc_tools_param.Parameters):
    def __init__(self, time_inter_seq, min_duration_intra_seq, min_len_seq, min_rep_nb,
                 max_branches, stop_if_twin, error_rate, no_reverse_seq, spike_rate_weight,
                 path_results=None, time_str=None, bin_size=1,
                 activity_threshold=None):
        """

        :param time_inter_seq: represent the maximum number of times between two elements of a sequences
        :param min_duration_intra_seq: represent the min number of times between two elements of a sequences,
        :param path_results:
        :param time_str:
        :param bin_size:
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

        # Tree parameters
        self.max_branches = max_branches
        self.stop_if_twin = stop_if_twin
        # making some change for gitkraken test

def build_mle_transition_dict(spike_nums, param, try_uniformity_method=False, debug_mode=False):
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

            # TODO: do something similar to Robin, to detect events, with percentile
            # doesn't change anything
            not_for_now = True
            if not not_for_now:
                if param.activity_threshold is not None:
                    if np.sum(spike_nums[:, t:t_max]) > param.activity_threshold:
                        print(f'spike_nums[:, t:t_max]) > {param.activity_threshold})')
                        continue
            actual_neurons_spikes = spike_nums[n, :] > 0
            spike_nums[n, actual_neurons_spikes] = 0

            pos = np.where(spike_nums[:, t:t_max])[0]
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

                # TODO: do something similar to Robin, to detect events, with percentile
                # doesn't change anything
                not_for_now = True
                if not not_for_now:
                    if param.activity_threshold is not None:
                        if np.sum(spike_nums[:, t:t_max]) > param.activity_threshold:
                            if debug_mode:
                                print(f'spike_nums[:, t:t_max]) > {param.activity_threshold})')
                            continue
                actual_neurons_spikes = spike_nums[n, :] > 0
                spike_nums[n, actual_neurons_spikes] = 0

                pos = np.where(spike_nums[:, t:t_max])[0]
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
        print(f'mean transition: {np.mean(transition_dict)}')
        print(f'std transition: {np.std(transition_dict)}')
        print(f'min transition: {np.min(transition_dict)}')
        print(f'max transition: {np.max(transition_dict)}')
    return transition_dict


def bfs(trans_dict, neuron_to_start, param):
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
            # threshold_prob =  (mean_trans_dict-std_trans_dict)
            threshold_prob = mean_trans_dict
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


def find_sequences(spike_nums, param, try_uniformity_method=False, debug_mode=False):
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
                                                debug_mode=debug_mode)

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
        for seq in sequences:
            # print(f"len seq in sequences: {len(seq)}")
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
                        if (nb_error < param.error_rate) and (not last_is_error.all()):
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
            # removing seq for which no instance have less than 2 errors comparing to probabilistic seq
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


def order_spike_nums_by_seq(spike_nums, param, with_printing=True, reverse_order=False):
    """

    :param spike_nums:
    :param param: instance of MarkovParameters
    :param with_printing:
    :return:
    """
    # ordered_spike_nums = spike_nums.copy()
    nb_neurons = len(spike_nums)
    if nb_neurons == 0:
        return [], []

    # list_seq_dict
    list_seq_dict, dict_by_len_seq, max_seq_dict = find_sequences(spike_nums=spike_nums, param=param)
    list_seq_dict_uniform, dict_by_len_seq_uniform, max_seq_dict_uniform = \
        find_sequences(spike_nums=spike_nums, param=param,
                       try_uniformity_method=True)

    # list_seq_dict: list of length number of neurons, each elemeent represent a dictionnary containing
    # the sequences beginnning by this neuron

    # first we build a dictionnary from the dictionnary list
    seq_dict = dict()
    for seq_dict_from_list in list_seq_dict:
        seq_dict.update(seq_dict_from_list)

    # adding sequences with algorithm taking into consideration un
    for seq_dict_from_list in list_seq_dict_uniform:
        seq_dict.update(seq_dict_from_list)

    # temporary code to test if taking the longest sequence is good option
    best_seq = None
    best_loss_score = 1
    for k, seq in max_seq_dict.items():
        seq = np.array(seq)
        new_order = np.zeros(nb_neurons, dtype="int8")
        new_order[:len(seq)] = seq
        # print(f"{k} max_seq seq len {len(seq)}")
        non_ordered_neurons = np.setdiff1d(np.arange(nb_neurons),
                                           seq)
        if len(non_ordered_neurons) > 0:
            # will update seq_dict with new seq if necessary and give back an ordered_neurons list
            ordered_neurons = order_cells_from_seq_dict(seq_dict=seq_dict, non_ordered_neurons=non_ordered_neurons,
                                                        param=param)
            # TODO: fin a way to order those not_ordered_neurons from list_seq_dict
            new_order[len(seq):] = ordered_neurons
            # new_order[len(seq):] = not_ordered_neurons
        tmp_spike_nums = spike_nums[new_order, :]
        loss_score = loss_function_with_sliding_window(spike_nums=tmp_spike_nums[::-1, :],
                                                       time_inter_seq=param.time_inter_seq,
                                                       min_duration_intra_seq=param.min_duration_intra_seq)
        if with_printing:
            print(f'loss_score neuron {k}, len {len(seq)}: {np.round(loss_score, 4)}')
        if loss_score < best_loss_score:
            best_loss_score = loss_score
            best_seq = np.array(new_order)

    if with_printing:
        print(f'best loss_score neuron {np.round(best_loss_score, 4)}')

    if with_printing:
        print("####### all sequences #######")
        for key, value in seq_dict.items():
            print(f"seq: {key}, rep: {len(value)}")

    result_seq_dict = dict()
    if best_seq is not None:

        selected_seq = get_seq_included_in(ref_seq=best_seq, seqs_to_test=list(seq_dict.keys()),
                                           min_len=2) # param.min_len_seq)
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
        ordered_result_seq_dict
    if reverse_order:
        # we are reversing best seq, so the seq will appear from top to bottom
        best_seq = best_seq[::-1]

    # if test_new_version:
    return ordered_result_seq_dict, best_seq
    # else:
    #     return list_seq_dict, best_seq


def get_seq_included_in(ref_seq, seqs_to_test, min_len=4):
    """
    Return all seq (list of tuple of int) from seqs_to_test that are in ref_seq (tuple of int), in the exact same order
    :param ref_seq:
    :param seq_to_test:
    :return:
    """
    result = []

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

    return result