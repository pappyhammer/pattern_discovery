import numpy as np
import pattern_discovery.tools.trains as trains_module


def loss_function_with_sliding_window(spike_nums, time_inter_seq, min_duration_intra_seq, spike_train_mode=False,
                                      debug_mode=False):
    """
    Return a float from 0 to 1, representing the loss function.
    If spike_nums is perfectly organized as sequences (meaning that for each spike of a neuron n, the following
    spikes of other neurons (on the next lines) are the same for each given spike of n.
    Sequences are supposed to go from neurons with max index to low index
    :param spike_nums: np.array of 2 dim, first one (lines) representing the neuron numbers and 2nd one the time.
    Binary array. If value at one, represents a spike.
    :param time_inter_seq: represent the maximum number of times between two elements of a sequences
    :param min_duration_intra_seq: represent the min number of times between two elements of a sequences,
    could be negative
    :param spike_train_mode: if True, then spike_nums should be a list of array, each list representing the spikes
    of a neuron
    :return:
    """
    loss = 0.0
    # size of the sliding matrix
    sliding_size = 3
    s_size = sliding_size
    nb_neurons = len(spike_nums)
    if nb_neurons <= s_size:
        return Exception(f"number of neurons are too low, min is {s_size+1}")
    if spike_train_mode:
        min_time, max_time = trains_module.get_range_train_list(spike_nums)
        rev_spike_nums = spike_nums[::-1]
    else:
        max_time = len(spike_nums[0, :])
        rev_spike_nums = spike_nums[::-1, :]

    # max time between 2 adjacent spike of a sequence
    # correspond as well to the highest (worst) loss for a spike
    max_time_inter = time_inter_seq - min_duration_intra_seq
    # nb_spikes_total = np.sum(spike_nums)
    if spike_train_mode:
        nb_spikes_used = 0
        for cell_index in np.arange(s_size, nb_neurons-s_size):
            # multiplying the number of spikes by sliding_window size * 2, in order to count the maximum number
            # of times a spike could be counted
            nb_spikes_used += len(spike_nums[cell_index]) * (s_size * 2)
        for i in np.arange(s_size):
            nb_spikes_used += (len(spike_nums[i]) + len(spike_nums[-(i + 1)])) * (i + 1)
    else:
        nb_spikes_used = np.sum(spike_nums[s_size:-s_size, :]) * (s_size * 2)
        for i in np.arange(s_size):
            nb_spikes_used += (np.sum(spike_nums[i, :]) + np.sum(spike_nums[-(i + 1), :])) * (i + 1)

    worst_loss = max_time_inter * nb_spikes_used
    # if debug_mode:
    #     print(f'nb_neurons {nb_neurons}, worst_loss {worst_loss}')

    for n, neuron in enumerate(rev_spike_nums):
        if n == (nb_neurons - (sliding_size + 1)):
            break

        if spike_train_mode:
            n_times = neuron
        else:
            n_times = np.where(neuron)[0]
        # next_n_times = np.where(rev_spike_nums[n+1, :])[0]
        # if len(n_times) == len(next_n_times):
        #     if np.all(np.diff(n_times) == np.diff(next_n_times)):
        #         continue

        # mask allowing to remove the spikes already taken in consideration to compute the loss
        # mask_next_n = np.ones((sliding_size, max_time_inter*sliding_size), dtype="bool")

        # will contain for each neuron of the sliding window, the diff value of each spike comparing to the first
        # neuron of the seq
        mean_diff = dict()
        for i in np.arange(1, sliding_size + 1):
            mean_diff[i] = []
        # we test for each spike of n the sliding_size following seq spikes
        for n_t in n_times:
            start_t = n_t + min_duration_intra_seq
            start_t = max(0, start_t)
            # print(f'start_t {start_t} max_time {max_time}')
            if (start_t + (max_time_inter * sliding_size)) < max_time:
                if spike_train_mode:
                    seq_mat = []
                    for cell_index in np.arange(n, (n + sliding_size + 1)):
                        cell_time_stamps = rev_spike_nums[cell_index]
                        end_t = start_t + (max_time_inter * sliding_size)
                        # selecting spikes in that interval [start_t:end_t]
                        spikes = cell_time_stamps[np.logical_and(cell_time_stamps >= start_t, cell_time_stamps < end_t)]
                        # copy might not be necessary, but just in case
                        seq_mat.append(np.copy(spikes))
                else:
                    seq_mat = np.copy(rev_spike_nums[n:(n + sliding_size + 1),
                                      start_t:(start_t + (max_time_inter * sliding_size))])
            else:
                if spike_train_mode:
                    seq_mat = []
                    for cell_index in np.arange(n, (n + sliding_size + 1)):
                        cell_time_stamps = rev_spike_nums[cell_index]
                        # selecting spikes in that interval [start_t:end_t]
                        spikes = cell_time_stamps[cell_time_stamps >= start_t]
                        # copy might not be necessary, but just in case
                        seq_mat.append(np.copy(spikes))
                else:
                    seq_mat = np.copy(rev_spike_nums[n:(n + sliding_size + 1),
                                  start_t:])
            # print(f'len(seq_mat) {len(seq_mat)} {len(seq_mat[0,:])}')
            # Keeping only one spike by neuron
            # indicate from which time we keep the first neuron, the neurons spiking before from_t are removed
            if spike_train_mode:
                from_t = min_time
            else:
                from_t = 0
            if spike_train_mode:
                first_neuron_t = seq_mat[0][0]
            else:
                first_neuron_t = np.where(seq_mat[0, :])[0][0]
            for i in np.arange(1, sliding_size + 1):
                if spike_train_mode:
                    bool_array = (seq_mat[i] >= from_t)
                    n_true = seq_mat[i][bool_array]
                    if len(n_true) > 1:
                        mask = np.ones(len(seq_mat[i]), dtype="bool")
                        value_sup_indices = np.where(bool_array)[0]
                        mask[value_sup_indices[1:]] = False
                        # removing all the spikes found after except the first one
                        seq_mat[i] = seq_mat[i][mask]
                else:
                    n_true = np.where(seq_mat[i, from_t:])[0]
                    # n_true is an array of int
                    if len(n_true) > 1:
                        # removing the spikeS after
                        n_true_min = n_true[1:]
                        seq_mat[i, n_true_min] = 0
                # removing the spikes before
                if from_t > 0:
                    if spike_train_mode:
                        bool_array = seq_mat[i] < from_t
                        t_before = seq_mat[i][bool_array]
                        if len(t_before) > 0:
                            mask = np.ones(len(seq_mat[i]), dtype="bool")
                            value_sup_indices = np.where(bool_array)[0]
                            mask[value_sup_indices] = False
                            # removing all the spikes found
                            seq_mat[i] = seq_mat[i][mask]
                    else:
                        t_before = np.where(seq_mat[i, :from_t])[0]
                        if len(t_before) > 0:
                            seq_mat[i, t_before] = 0
                if len(n_true > 0):
                    # keeping the diff between the spike of the neuron in position i from the neuron n first spike
                    mean_diff[i].append((n_true[0] + from_t) - first_neuron_t)
                    from_t = n_true[0] + min_duration_intra_seq
                    from_t = max(0, from_t)
            # seq_mat is not used so far, but could be used for another technique.
        # we add to the loss_score, the std of the diff between spike of the first neuron and the other
        for i in np.arange(1, sliding_size + 1):
            if len(mean_diff[i]) > 0:
                # print(f'Add loss mean {np.mean(mean_diff[i])}, std {np.std(mean_diff[i])}')
                loss += min(np.std(mean_diff[i]), max_time_inter)
            # then for each spike of the neurons not used in the diff, we add to the loss the max value = max_time_inter
            # print(f'n+i {n+i} len(np.where(rev_spike_nums[n+i, :])[0]) {len(np.where(rev_spike_nums[n+i, :])[0])}'
            #       f' len(mean_diff[i]) {len(mean_diff[i])}')

            # nb_not_used_spikes could be zero when a neuron spikes a lot, and the following one not so much
            # then spikes from the following one will be involved in more than one seq
            if spike_train_mode:
                nb_not_used_spikes = max(0, (len(rev_spike_nums[n + i]) - len(mean_diff[i])))
            else:
                nb_not_used_spikes = max(0, (len(np.where(rev_spike_nums[n + i, :])[0]) - len(mean_diff[i])))
            # if nb_not_used_spikes < 0:
            #     print(f'ERROR: nb_not_used_spikes inferior to 0 {nb_not_used_spikes}')
            # else:
            loss += nb_not_used_spikes * max_time_inter
        # print(f"loss_score n {loss}")

    # loss should be between 0 and 1
    return loss / worst_loss
