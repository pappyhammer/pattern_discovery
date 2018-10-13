import numpy as np
import numpy.random as rnd
import pattern_discovery.tools.trains as trains_module
import math


# TODO: create dithered method for list of spike_trains, using code and book from Drun
def jitter_spike_train(train, sigma):
    """
        Compute new spike trains jitterd by gaussian (centered) noise with sd  sigma
        train should be in the form train[i] = T ith spike occurs at time T
    :param train:
    :param sigma:
    :return: new spike trains - with same number of spike len = len(train)

    """

    n = len(train)
    new_train = train.copy()
    new_train += sigma * rnd.randn(n)
    return new_train


def dithered_data_set(train_list, min_value, max_value):
    """
        create new train list with jittered spike trains
    :param train_list: the spike train list to be jittered
    :param sigma: noise of jittering
    :return: new jittered train list

    """
    jittered_list = []
    for train in train_list:
        new_train = trains_module.shifted(train, shift=(max_value-min_value), rng=(min_value, max_value))
        jittered_list.append(new_train)
    return jittered_list


def create_surrogate_dataset(train_list, nsurrogate, min_value, max_value):
    """

    :param train_list:
    :param nsurrogate:
    :param sigma: noise of jittering
    :return:
    """
    surrogate_data_set = []
    for i in range(nsurrogate):
        surrogate_data_set.append(dithered_data_set(train_list, min_value, max_value))
    return surrogate_data_set


def get_sce_detection_threshold(spike_nums, window_duration, n_surrogate, perc_threshold=95, non_binary=False,
                                debug_mode=False, spike_train_mode=False, sigma=0.1):
    """
    Compute the activity threshold (ie: nb of onset at each time, if param.bin_size > 1, then will first bin
    the spike by bin_size times then compute the threshold.
    :param spike_nums:
    :param non_binary: means that spike_nums could hold values that are not only 0 or 1
    :param spike_train_mode: if True, spike_nums should be a list of np.array with float or int value
    representing the spike time of each cell (each np.array representing a cell)
    :return:
    """
    if debug_mode:
        print("start get activity threshold")

    if spike_train_mode:
        min_time, max_time = trains_module.get_range_train_list(spike_nums)
        surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
                                                      min_value=min_time, max_value=max_time)
        n_times = int(math.ceil(max_time - min_time))
        n_cells = len(spike_nums)
        just_keeping_the_max_of_each_surrogate = False

        number_of_times_without_spikes = 0

        if just_keeping_the_max_of_each_surrogate:
            n_rand_sum = []
        else:
            n_rand_sum = np.zeros(0)

        if debug_mode:
            print(f"window_duration {window_duration}")
        for i, surrogate_train_list in enumerate(surrogate_data_set):
            if debug_mode:
                if (i % 5) == 0:
                    print(f"surrogate n°: {i}")
            # to make it faster, we keep the count of cells in a dict, thus not having to create a huge
            # matrix if only a sparse number of times have spikes
            # this dict will have as key the cell number and as value a set containing
            # the time in wich a spike was counting as part of active during a window
            # using a set allows to keep it simple and save computational time (hopefully)
            windows_set = dict()
            for cell_number in np.arange(n_cells):
                windows_set[cell_number] = set()

            for cell, spikes_train in enumerate(spike_nums):
                # if debug_mode and (cell == 0):
                #     print(f"len(spikes_train): {len(spikes_train)}")
                for spike_time in spikes_train:
                    # first determining to which windows to add the spike
                    spike_index = int(spike_time - min_time)
                    first_index_window = np.max((0, int(spike_index - window_duration)))
                    # we add to the set of the cell, all indices in this window
                    windows_set[cell].update(np.arange(first_index_window, spike_index))

            # uint8 : int from 0 to 255
            # max sum should be n_cells
            # for memory optimization
            if n_cells < 255:
                count_array = np.zeros(n_times, dtype="uint8")
            else:
                count_array = np.zeros(n_times, dtype="uint16")
            for cell, times in windows_set.items():
                times = np.asarray(list(times))
                # mask = np.zeros(n_times, dtype="bool")
                # mask[times] = True
                count_array[times] = count_array[times] + 1

            # print("after windows_sum")
            sum_spikes = count_array[count_array>0]
            # not to have to keep a huge array, we just keep values superior to 0 and we keep the count
            # off how many times are at 0
            number_of_times_without_spikes += (n_times - (len(count_array) - len(sum_spikes)))
            # concatenating the sum of spikes for each time
            if just_keeping_the_max_of_each_surrogate:
                n_rand_sum.append(np.max(sum_spikes))
            else:
                n_rand_sum = np.concatenate((n_rand_sum, sum_spikes))

        if just_keeping_the_max_of_each_surrogate:
            n_rand_sum = np.asarray(n_rand_sum)
        else:
            # if debug_mode:
            #     print(f"number_of_times_without_spikes {number_of_times_without_spikes}")
            n_rand_sum = np.concatenate((n_rand_sum, np.zeros(number_of_times_without_spikes, dtype="uint16")))
            pass

        activity_threshold = np.percentile(n_rand_sum, perc_threshold)

        return activity_threshold

    # ------------------- for non spike_train_mode ------------------

    if non_binary:
        binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
        for neuron, spikes in enumerate(spike_nums):
            binary_spikes[neuron, spikes > 0] = 1
        spike_nums = binary_spikes

    n_times = len(spike_nums[0, :])

    # computing threshold to detect synchronous peak of activity
    n_rand_sum = np.zeros(n_surrogate * n_times)
    for i in np.arange(n_surrogate):
        if debug_mode:
            print(f"surrogate n°: {i}")
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
        max_sum = 0
        for t in np.arange(0, (n_times - window_duration)):
            sum_value = np.sum(spike_nums[:, t:(t + window_duration)])
            n_rand_sum[(i * n_times) + t] = sum_value
            # max_sum = max(sum_value, max_sum)
        for t in np.arange((n_times - window_duration), n_times):
            sum_value = np.sum(spike_nums[:, t:])
            n_rand_sum[(i * n_times) + t] = sum_value
        # Keeping the max value for each surrogate data
        # n_rand_sum[i] = max_sum

    activity_threshold = np.percentile(n_rand_sum, perc_threshold)

    return activity_threshold

# TODO: same method but with spike_trains
# TODO: for concatenation of SCE, if the same cells spike more than one, then the following should be considered
# in the count of cells active after the first SCE
def detect_sce_with_sliding_window(spike_nums, window_duration, perc_threshold=95,
                                   with_refractory_period=-1, non_binary=False,
                                   activity_threshold=None, debug_mode=False,
                                   no_redundancy=False):

    """
    Use a sliding window to detect sce (define as peak of activity > perc_threshold percentile after
    randomisation during a time corresponding to window_duration)
    :param spike_nums: 2D array, lines=cells, columns=time
    :param window_duration:
    :param perc_threshold:
    :param no_redundancy: if True, then when using the sliding window, a second spike of a cell is not taking into
    consideration when looking for a new SCE
    :return: ** one array (mask, boolean) containing True for indices (times) part of an SCE,
    ** a list of tuple corresponding to the first and last index of each SCE, (last index being included in the SCE)
    ** sce_nums: a new spike_nums with in x axis the SCE and in y axis the neurons, with 1 if
    active during a given SCE.
    ** an array of len n_times, that for each times give the SCE number or -1 if part of no cluster
    ** activity_threshold

    """

    if non_binary:
        binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
        for neuron, spikes in enumerate(spike_nums):
            binary_spikes[neuron, spikes > 0] = 1
        spike_nums = binary_spikes

    if activity_threshold is None:
        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums, n_surrogate=100,
                                                         window_duration=window_duration,
                                                         perc_threshold=perc_threshold,
                                                         non_binary=False)

    n_cells = len(spike_nums)
    n_times = len(spike_nums[0, :])
    start_sce = -1
    # keep a trace of which cells have been added to an SCE
    cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
    sce_bool = np.zeros(n_times, dtype="bool")
    sce_tuples = []
    sce_times_numbers = np.ones(n_times, dtype="int16")
    sce_times_numbers *= -1
    if debug_mode:
        print(f"n_times {n_times}")
    for t in np.arange(0, (n_times - window_duration)):
        if debug_mode:
            if t % 10**6 == 0:
                print(f"t {t}")
        cells_has_been_removed_due_to_redundancy = False
        sum_value_test = np.sum(spike_nums[:, t:(t + window_duration)])
        sum_spikes = np.sum(spike_nums[:, t:(t + window_duration)], axis = 1)
        pos_cells = np.where(sum_spikes)[0]
        # neurons with sum > 1 are active during a SCE
        sum_value = len(pos_cells)
        if no_redundancy and (start_sce > -1):
            # removing from the count the cell that are in the previous SCE
            nb_cells_already_in_sce = np.sum(cells_in_sce_so_far[pos_cells])
            sum_value -= nb_cells_already_in_sce
            if nb_cells_already_in_sce > 0:
                cells_has_been_removed_due_to_redundancy = True
        # print(f"Sum value, test {sum_value_test}, rxeal {sum_value}")
        if sum_value > activity_threshold:
            if start_sce == -1:
                start_sce = t
                if no_redundancy:
                    # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                    sum_spikes = np.sum(spike_nums[:, t])
                    pos_cells = np.where(sum_spikes)[0]
                    cells_in_sce_so_far[pos_cells] = True
            else:
                if no_redundancy:
                    # updating which cells are already in the SCE
                    # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                    sum_spikes = np.sum(spike_nums[:, t])
                    pos_cells = np.where(sum_spikes)[0]
                    cells_in_sce_so_far[pos_cells] = True
                else:
                    pass
        else:
            if start_sce > -1:
                # then a new SCE is detected
                sce_bool[start_sce:t] = True
                # sce_tuples.append((start_sce, (t + window_duration) - 2))
                sce_tuples.append((start_sce, t-1))
                sce_times_numbers[start_sce:t] = len(sce_tuples) - 1
                start_sce = -1
                cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
            if no_redundancy and cells_has_been_removed_due_to_redundancy:
                sum_value += nb_cells_already_in_sce
                if sum_value > activity_threshold:
                    # then a new SCE start right after the old one
                    start_sce = t
                    if no_redundancy:
                        # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                        sum_spikes = np.sum(spike_nums[:, t])
                        pos_cells = np.where(sum_spikes)[0]
                        cells_in_sce_so_far[pos_cells] = True

    n_sces = len(sce_tuples)
    sce_nums = np.zeros((n_cells, n_sces), dtype="int16")
    for sce_index, sce_tuple in enumerate(sce_tuples):
        cells_spikes = np.zeros(n_cells, dtype="int8")
        sum_spikes = np.sum(spike_nums[:, sce_tuple[0]:(sce_tuple[1] + 1)], axis=1)
        # neurons with sum > 1 are active during a SCE
        active_cells = np.where(sum_spikes)[0]
        sce_nums[active_cells, sce_index] = 1

    # print(f"number of sce {len(sce_tuples)}")

    return sce_bool, sce_tuples, sce_nums, sce_times_numbers, activity_threshold


def compute_sce_threshold_from_raster_dur(spike_nums, perc_threshold=95, n_surrogate=1000, non_binary=False):
    if non_binary:
        binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
        for neuron, spikes in enumerate(spike_nums):
            binary_spikes[neuron, spikes > 0] = 1
        spike_nums = binary_spikes
    n_times = len(spike_nums[0, :])

    # computing threshold to detect synchronous peak of activity
    n_rand_sum = np.zeros(n_surrogate * n_times)
    for i in np.arange(n_surrogate):
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))

        count = np.sum(copy_spike_nums, axis=0)
        n_rand_sum[i * n_times:(i + 1) * n_times] = count

    sce_threshold = np.percentile(n_rand_sum, perc_threshold)

    return sce_threshold
