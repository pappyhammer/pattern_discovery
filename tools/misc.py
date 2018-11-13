import numpy as np

def get_spike_rates(spike_nums):
    """
    Give the spike rate for each neuron (float between 0 and 1)
    :param spike_nums:
    :return:
    """
    (nb_neurons, nb_time) = spike_nums.shape
    spike_rates = np.zeros(nb_neurons)
    for n, neuron in enumerate(spike_nums):
        spike_rates[n] = np.sum(neuron) / nb_time
    return spike_rates


def get_spikes_duration_from_raster_dur(spike_nums_dur):
    spike_durations = []
    for cell_id, spikes_time in enumerate(spike_nums_dur):
        if len(spikes_time) == 0:
            spike_durations.append([])
            continue
        n_times = len(spikes_time)
        d_times = np.diff(spikes_time)
        # show the +1 and -1 edges
        pos = np.where(d_times == 1)[0] + 1
        neg = np.where(d_times == -1)[0] + 1

        if (pos.size == 0) and (neg.size == 0):
            if len(np.nonzero(spikes_time)[0]) > 0:
                spike_durations.append([n_times])
            else:
                spike_durations.append([])
        elif pos.size == 0:
            # i.e., starts on an spike, then stops
            spike_durations.append([neg[0]])
        elif neg.size == 0:
            # starts, then ends on a spike.
            spike_durations.append([n_times - neg[0]])
        else:
            if pos[0] > neg[0]:
                # we start with a spike
                pos = np.insert(pos, 0, 0)
            if neg[-1] < pos[-1]:
                #  we end with aspike
                neg = np.append(neg, n_times - 1)
            # NOTE: by this time, length(pos)==length(neg), necessarily
            h = np.matrix([pos, neg])
            if np.any(h):
                goodep = np.array(h[1, :] - h[0, :]).flatten()
                spike_durations.append(list(goodep))

    return spike_durations


def get_time_correlation_data(spike_nums, events_times):
    """
       Will compute data that will be use in order to plot the time-correlation graph
       :return:
    """
    # ms_scale represents the space between each tick
    nb_neurons = len(spike_nums)
    n_times = len(spike_nums[0, :])
    # values for each cell
    time_lags_dict = dict()
    correlation_dict = dict()
    # for ploting
    time_lags_list = []
    correlation_list = []

    # first determining what is the maximum duration of an event, for array dimension purpose
    max_duration_event = 0
    for times in events_times:
        max_duration_event = np.max((max_duration_event, times[1]-times[0]))

    time_window = int(np.ceil(max_duration_event / 2))

    for neuron in np.arange(nb_neurons):
        # look at onsets
        neuron_spikes, = np.where(spike_nums[neuron, :])

        if len(neuron_spikes) == 0:
            continue

        spike_nums_to_use = spike_nums

        # time_window by 4
        distribution_array_2_d = np.zeros((nb_neurons, ((time_window * 4) + 1)),
                                          dtype="int16")

        mask = np.ones(nb_neurons, dtype="bool")
        mask[neuron] = False

        # event_index = time_window
        # looping on each spike of the main neuron
        for n, event_times in enumerate(events_times):
            # only taking in consideration events that are not too close from bottom range or upper range
            min_limit = event_times[0]
            max_limit = min(event_times[1]+1, (n_times - 1)) # min((peak_time + time_window), (n_times - 1))
            if np.sum(spike_nums[neuron, min_limit:max_limit]) == 0:
                continue
            # see to consider the case in which the cell spikes 2 times around a peak during the tim_window
            neuron_spike_time = np.where(spike_nums[neuron, min_limit:max_limit])[0][0]
            spikes_indices = np.where(spike_nums_to_use[:, min_limit:max_limit])
            conn_cells_indices = spikes_indices[0]
            spikes_indices = neuron_spike_time - spikes_indices[1]
            spikes_indices += time_window*2
            # print(f"spikes_indices {spikes_indices}")
            # copy_of_neuron_distrib = np.copy(distribution_array_2_d[neuron, :])
            distribution_array_2_d[conn_cells_indices, spikes_indices] += 1
            # distribution_array_2_d[neuron, :] = copy_of_neuron_distrib

        # sum of spikes at each times lag
        distribution_array = np.sum(distribution_array_2_d[mask, :], axis=0)
        # print(f"distribution_array {distribution_array}")
        total_spikes = np.sum(distribution_array)
        # adding the cell only if it has at least a spike around peak times
        if total_spikes > 0:
            correlation_value = np.max(distribution_array) / total_spikes
            # array_to_average = np.zeros(np.sum(distribution_array))
            # start = 0
            # for index, time_lag in enumerate(np.arange(-time_window * 2, time_window * 2 + 1)):
            #     n_spike_for_this_time_lag = distribution_array[index]
            #     array_to_average[start:(start+n_spike_for_this_time_lag)] = time_lag
            #     start += n_spike_for_this_time_lag
            # avg_time_lag = np.mean(array_to_average)
            # other way:
            time_lags_range = np.arange(-time_window * 2, time_window * 2 + 1)
            distribution_array = distribution_array * time_lags_range
            avg_time_lag = np.sum(distribution_array)/total_spikes
            time_lags_dict[neuron] = avg_time_lag
            correlation_dict[neuron] = correlation_value

    for cell, time_lag in time_lags_dict.items():
        time_lags_list.append(time_lag)
        correlation_list.append(correlation_dict[cell])

    return time_lags_list, correlation_list, time_lags_dict, correlation_dict, time_window


def get_isi(spike_data, spike_trains_format=False):
    """

    :param spike_data:
    :param spike_trains_format:
    :return: a dict with as key an int representing the cell index, and as value a list of value representing the
    interspike interval between each spike of the cell
    """

    isi_by_neuron = dict()
    for cell, spikes in enumerate(spike_data):
        if not spike_trains_format:
            spikes, = np.where(spikes)
        isi = np.diff(spikes)
        isi_by_neuron[cell] = isi

    return isi_by_neuron

def check_seq_for_neurons(spike_nums, param, neurons_to_checks):
    """

    :param spike_nums:
    :param param:
    :param keep_inter_seq:
    :param neurons_to_checks: np.array
    :return:
    """
    nb_rep = 0
    index_list = []
    cells_list = []
    # TODO: complete it
    return nb_rep, cells_list, index_list


def find_seq_in_ordered_raster(spike_nums, param, keep_inter_seq):
    """
    Will find in an order spike_nums(cells being the raws, and the spikes in columns), cells being ordered from N_cells
    to zero, sequences of cells with a minimum len and rep (from param).
    :param spike_nums:
    :param param: Instance of Markov paramaters. min_seq_len, min_seq_len and error_rate will be used.
    :param keep_inter_seq: if True, then a shorter cells seq will be return even so it's included in a longer one.
    :return: three list: first one being the list of sequences of cells as tuples (indices), second one is a dict with
    key a tuple representing a seq of cells and the value is a list of tuple: each tuple containing a list of cells
    indices and the other list the corresponsding spikes indices (time).
    """
    neuron_index = len(spike_nums)-1
    # list of tuples
    selected_seq = []
    selection_dict = dict()
    actual_min_len_seq = param.min_len_seq
    while neuron_index >= (actual_min_len_seq-1):
        # we first hypothetize that cells from neuron_index to (neuron_index - param.min_len_seq) forms a seq with a min
        # repetition equel to param.min_rep_nb. If that's so, then we'll try to extend it as much as possible and
        # we will keep the different versions
        neurons_to_checks = np.arange(neuron_index, (neuron_index-param.min_len_seq), -1)
        nb_rep, cells_list, index_list = check_seq_for_neurons(spike_nums=spike_nums, param=param,
                                                               neurons_to_checks=neurons_to_checks)
        if nb_rep < param.min_rep_nb:
            neuron_index -= 1
            actual_min_len_seq = param.min_len_seq
            continue

        # before adding seq, we need to check if it's not included in another already added seq
        # if keep_inter_seq is False:
        if not keep_inter_seq:
            # TODO: keep_inter_seq
            pass
        selected_seq.append(tuple(neurons_to_checks))
        selection_dict[tuple(neurons_to_checks)] = [cells_list, index_list]
        actual_min_len_seq += 1

    return


def find_continuous_frames_period(frames):
    """
    Take an array of frames number, and return a list of tuple corresponding to beginning and end of series of frames
    that are continuous
    :param frames:
    :return:
    """
    if len(frames) == 0:
        return []
    diff_frames = np.diff(frames)
    end_edges = np.where(diff_frames > 1)[0]
    end_edges = np.append(end_edges, len(diff_frames))
    frames_periods = []
    edges = []
    i = 0
    while i < len(end_edges):
        index = end_edges[i]
        if i == 0:
            frames_periods.append((frames[0], frames[index]))
            edges.append(frames[0])
        else:
            previous_index = end_edges[i - 1]
            frames_periods.append((frames[previous_index + 1], frames[index]))
            edges.append(frames[previous_index + 1])
        edges.append(frames[index])
        i += 1
    return frames_periods