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