import numpy as np
import numpy.random as rnd
import pattern_discovery.tools.trains as trains_module
import math
from pattern_discovery.tools.misc import get_continous_time_periods
from matplotlib import pyplot as plt
from pattern_discovery.tools.signal import smooth_convolve
from scipy import signal


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


def get_sce_detection_threshold(spike_nums, window_duration, n_surrogate, use_max_of_each_surrogate=False,
                                perc_threshold=95, non_binary=False,
                                debug_mode=False, spike_train_mode=False):
    """
    Compute the activity threshold (ie: nb of onset at each time, if param.bin_size > 1, then will first bin
    the spike by bin_size times then compute the threshold.
    :param spike_nums:
    :param non_binary: means that spike_nums could hold values that are not only 0 or 1
    :param spike_train_mode: if True, spike_nums should be a list of np.array with float or int value
    representing the spike time of each cell (each np.array representing a cell)
    :param use_max_of_each_surrogate: if True, the percentile threshold will be applied to the max sum of each
    surrogate generated.
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
        just_keeping_the_max_of_each_surrogate = use_max_of_each_surrogate

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

            for cell, spikes_train in enumerate(surrogate_train_list):
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
    if use_max_of_each_surrogate:
        n_rand_sum = np.zeros(n_surrogate)
    else:
        if window_duration == 1:
            n_rand_sum = np.zeros(n_surrogate * n_times )
        else:
            n_rand_sum = np.zeros(n_surrogate * (n_times-window_duration))
    for i in np.arange(n_surrogate):
        if debug_mode:
            print(f"surrogate n°: {i}")
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))
        if window_duration == 1:
            n_rand_sum[i*n_times:(i+1)*n_times] = np.sum(copy_spike_nums, axis=0)
            continue
        max_sum = 0
        for t in np.arange(0, (n_times - window_duration)):
            sum_value = np.sum(copy_spike_nums[:, t:(t + window_duration)])
            max_sum = np.max((sum_value, max_sum))
            if not use_max_of_each_surrogate:
                n_rand_sum[(i * (n_times - window_duration)) + t] = sum_value

        # for t in np.arange((n_times - window_duration), n_times):
        #     sum_value = np.sum(spike_nums[:, t:])
        #     n_rand_sum[(i * n_times) + t] = sum_value

        # Keeping the max value for each surrogate data
        if use_max_of_each_surrogate:
            n_rand_sum[i] = max_sum

    activity_threshold = np.percentile(n_rand_sum, perc_threshold)

    return activity_threshold


def get_low_activity_events_detection_threshold(spike_nums, window_duration, n_surrogate,
                                                 use_min_of_each_surrogate=False,
                                                perc_threshold=5, non_binary=False,
                                                debug_mode=False, spike_train_mode=False):
    """
    Compute the low_activity events threshold
    :param spike_nums:
    :param non_binary: means that spike_nums could hold values that are not only 0 or 1
    :param spike_train_mode: if True, spike_nums should be a list of np.array with float or int value
    representing the spike time of each cell (each np.array representing a cell)
    :param use_min_of_each_surrogate: if True, the percentile threshold will be applied to the min sum of each
    surrogate generated.
    :return:
    """
    if debug_mode:
        print("start get low activity events threshold")

    if spike_train_mode:
        min_time, max_time = trains_module.get_range_train_list(spike_nums)
        surrogate_data_set = create_surrogate_dataset(train_list=spike_nums, nsurrogate=n_surrogate,
                                                      min_value=min_time, max_value=max_time)
        n_times = int(math.ceil(max_time - min_time))
        n_cells = len(spike_nums)

        number_of_times_without_spikes = 0

        if use_min_of_each_surrogate:
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

            for cell, spikes_train in enumerate(surrogate_train_list):
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
            if use_min_of_each_surrogate:
                n_rand_sum.append(np.min(sum_spikes))
            else:
                n_rand_sum = np.concatenate((n_rand_sum, sum_spikes))

        if use_min_of_each_surrogate:
            n_rand_sum = np.asarray(n_rand_sum)
        else:
            # if debug_mode:
            #     print(f"number_of_times_without_spikes {number_of_times_without_spikes}")
            n_rand_sum = np.concatenate((n_rand_sum, np.zeros(number_of_times_without_spikes, dtype="uint16")))

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
    if use_min_of_each_surrogate:
        n_rand_sum = np.zeros(n_surrogate)
    else:
        n_rand_sum = np.zeros(n_surrogate * (n_times - window_duration))
    for i in np.arange(n_surrogate):
        if debug_mode:
            print(f"surrogate n°: {i}")
        copy_spike_nums = np.copy(spike_nums)
        for n, neuron_spikes in enumerate(copy_spike_nums):
            # roll the data to a random displace number
            copy_spike_nums[n, :] = np.roll(neuron_spikes, np.random.randint(1, n_times))

        min_sum = 0
        for t in np.arange(0, (n_times - window_duration)):
            sum_value = np.sum(copy_spike_nums[:, t:(t + window_duration)])
            min_sum = np.min((sum_value, min_sum))
            if not use_min_of_each_surrogate:
                n_rand_sum[(i * (n_times - window_duration)) + t] = sum_value

        # for t in np.arange((n_times - window_duration), n_times):
        #     sum_value = np.sum(spike_nums[:, t:])
        #     n_rand_sum[(i * n_times) + t] = sum_value

        # Keeping the max value for each surrogate data
        if use_min_of_each_surrogate:
            n_rand_sum[i] = min_sum

    activity_threshold = np.percentile(n_rand_sum, perc_threshold)

    return activity_threshold


def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def detect_sce_potatoes_style(spike_nums, perc_threshold=95, debug_mode=False, keep_only_the_peak=False):
    sum_spike_nums = np.sum(spike_nums, axis=0)
    sum_spike_nums_copy = np.copy(sum_spike_nums)
    sum_spike_nums_bin = np.copy(sum_spike_nums)
    n_times = sum_spike_nums_bin.shape[0]
    n_cells = spike_nums.shape[0]
    # smoothing the trace
    windows = ['hanning', 'hamming', 'bartlett', 'blackman']
    i_w = 1
    window_length = 11
    smooth_signal = smooth_convolve(x=sum_spike_nums, window_len=window_length,
                                        window=windows[i_w])
    beg = (window_length - 1) // 2
    sum_spike_nums= smooth_signal[beg:-beg]

    # binning
    bin_size = 100
    for i in np.arange(sum_spike_nums_bin.shape[0]):
        if i >= sum_spike_nums_bin.shape[0] - bin_size:
            break
        sum_spike_nums_bin[i] = np.mean(sum_spike_nums_bin[i:i+bin_size])
    smooth_signal = smooth_convolve(x=sum_spike_nums_bin, window_len=window_length,
                                    window=windows[i_w])
    beg = (window_length - 1) // 2
    sum_spike_nums_bin = smooth_signal[beg:-beg]
    mean_sum_spike_nums_bin = np.mean(sum_spike_nums_bin)
    std_sum_spike_nums_bin = np.std(sum_spike_nums_bin)

    # sum_spike_nums_bin = smooth_curve(sum_spike_nums_bin, factor=0.8)

    # finding SCE
    peak_nums = np.zeros(n_times, dtype="uint8")

    # using find_peaks
    # looking for peaks over mean
    height = np.mean(sum_spike_nums_bin)  # + np.std(traces[cell])
    peaks, properties = signal.find_peaks(x=sum_spike_nums_bin, distance=bin_size, height=height)
    # print(f"peaks {peaks}")
    peak_nums[peaks] = 1
    for peak_index in np.arange(len(peaks)):
        if peak_index == len(peaks) - 1:
            break
        if peaks[peak_index] == 0:
            continue
        if np.min(sum_spike_nums_bin[peaks[peak_index]:peaks[peak_index+1]+1]) >= mean_sum_spike_nums_bin:
            # then we keep the peak with the highest value
            if sum_spike_nums_bin[peaks[peak_index]] > sum_spike_nums_bin[peaks[peak_index+1]]:
                peak_nums[peaks[peak_index+1]] = 0
            else:
                peak_nums[peaks[peak_index]] = 0
    peak_nums[-bin_size:] = 0
    peaks = np.where(peak_nums)[0]

    sce_tuples = []
    sce_bool = np.zeros(n_times, dtype="bool")
    sce_times_numbers = np.ones(n_times, dtype="int16")
    sce_times_numbers *= -1
    end_sce_threshold = mean_sum_spike_nums_bin-(1.5*std_sum_spike_nums_bin)
    # then for each peak we decide when the "SCE" start and finish
    for peak_index in np.arange(len(peaks)):
        peak = peaks[peak_index]
        # start_sce = None
        if peak_index == 0:
            first_index = 0
            last_index = peaks[peak_index + 1]
        elif peak_index == len(peaks) -1:
            first_index = peaks[peak_index - 1]
            last_index = n_times - bin_size
        else:
            first_index = peaks[peak_index - 1]
            last_index = peaks[peak_index + 1]

        if np.min(sum_spike_nums_bin[first_index:peak]) < mean_sum_spike_nums_bin:
            start_sce = first_index + np.where(sum_spike_nums_bin[first_index:peak] < mean_sum_spike_nums_bin)[0][-1]
        else:
            start_sce = first_index + np.argmin(sum_spike_nums_bin[first_index:peak])

        if np.min(sum_spike_nums_bin[peak:last_index]) < end_sce_threshold:
            end_sce = peak + \
                      np.where(sum_spike_nums_bin[peak:last_index] < end_sce_threshold)[0][0]
        else:
            min_value = np.min(sum_spike_nums_bin[peak:last_index])
            end_sce = peak + np.where(sum_spike_nums_bin[peak:last_index] == min_value)[0][-1]

        sce_tuples.append((start_sce, end_sce))

    if keep_only_the_peak:
        new_sce_tuples = []
        for sce_index, sce_tuple in enumerate(sce_tuples):
            index_max = np.argmax(np.sum(spike_nums[:, sce_tuple[0]:sce_tuple[1] + 1], axis=0))
            new_sce_tuples.append((sce_tuple[0] + index_max, sce_tuple[0] + index_max))
        sce_tuples = new_sce_tuples
    for sce_index, sce_tuple in enumerate(sce_tuples):
        sce_bool[sce_tuple[0]:sce_tuple[1] + 1] = True
        sce_times_numbers[sce_tuple[0]:sce_tuple[1] + 1] = sce_index

    n_sces = len(sce_tuples)
    sce_nums = np.zeros((n_cells, n_sces), dtype="int16")
    for sce_index, sce_tuple in enumerate(sce_tuples):
        sum_spikes = np.sum(spike_nums[:, sce_tuple[0]:(sce_tuple[1] + 1)], axis=1)
        # neurons with sum > 1 are active during a SCE
        active_cells = np.where(sum_spikes)[0]
        sce_nums[active_cells, sce_index] = 1

    # print(f"number of sce {len(sce_tuples)}")
    show_plot = False
    if show_plot:
        # plt.plot(sum_spike_nums, zorder=8, color="black", alpha=0.4)
        plt.plot(sum_spike_nums_bin, zorder=10, color="blue", alpha=1, lw=3)
        plt.plot(sum_spike_nums_copy, color="red", alpha=0.4, zorder=7)
        plt.hlines(np.mean(sum_spike_nums_bin), 0, n_times-1, color="black",
                                         linewidth=0.5,
                                         linestyles="dashed")
        plt.hlines(end_sce_threshold, 0, n_times-1, color="red",
                                         linewidth=0.5,
                                         linestyles="dashed")
        size_peak_scatter = 50
        plt.scatter(peaks, sum_spike_nums_bin[peaks],
                    marker='o', c="yellow",
                    edgecolors="black", s=size_peak_scatter,
                    zorder=11, alpha=0.8)
        # print(f"sce_tuples {sce_tuples}")
        for index, coord in enumerate(sce_tuples):
            color = "black"
            # print(f"coord {coord}")
            plt.axvspan(coord[0], coord[1]+1, alpha=0.5, facecolor=color, zorder=1)
        plt.show()

    # raise Exception("TOTO TOTO")

    return sce_bool, sce_tuples, sce_nums, sce_times_numbers


# TODO: same method but with spike_trains
# TODO: for concatenation of SCE, if the same cells spike more than one, then the following should be considered
# in the count of cells active after the first SCE
def detect_sce_with_sliding_window(spike_nums, window_duration, perc_threshold=95,
                                   with_refractory_period=-1, non_binary=False,
                                   activity_threshold=None, debug_mode=False,
                                   no_redundancy=False, keep_only_the_peak=False):

    """
    Use a sliding window to detect sce (define as peak of activity > perc_threshold percentile after
    randomisation during a time corresponding to window_duration)
    :param spike_nums: 2D array, lines=cells, columns=time
    :param window_duration:
    :param perc_threshold:
    :param no_redundancy: if True, then when using the sliding window, a second spike of a cell is not taking into
    consideration when looking for a new SCE
    :param keep_only_the_peak: keep only the frame with the maximum cells co-activating
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
        activity_threshold = get_sce_detection_threshold(spike_nums=spike_nums, n_surrogate=1000,
                                                         window_duration=window_duration,
                                                         perc_threshold=perc_threshold,
                                                         non_binary=False)

    n_cells = len(spike_nums)
    n_times = len(spike_nums[0, :])

    if window_duration == 1:
        # using a diff method
        sum_spike_nums = np.sum(spike_nums, axis=0)
        binary_sum = np.zeros(n_times, dtype="int8")
        binary_sum[sum_spike_nums >= activity_threshold] = 1
        sce_tuples = get_continous_time_periods(binary_sum)
        if keep_only_the_peak:
            new_sce_tuples = []
            for sce_index, sce_tuple in enumerate(sce_tuples):
                index_max = np.argmax(np.sum(spike_nums[:, sce_tuple[0]:sce_tuple[1] + 1], axis=0))
                new_sce_tuples.append((sce_tuple[0]+index_max, sce_tuple[0]+index_max))
            sce_tuples = new_sce_tuples
        sce_bool = np.zeros(n_times, dtype="bool")
        sce_times_numbers = np.ones(n_times, dtype="int16")
        sce_times_numbers *= -1
        for sce_index, sce_tuple in enumerate(sce_tuples):
            sce_bool[sce_tuple[0]:sce_tuple[1]+1] = True
            sce_times_numbers[sce_tuple[0]:sce_tuple[1]+1] = sce_index

    else:
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
            if sum_value >= activity_threshold:
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
                    if keep_only_the_peak:
                        index_max = np.argmax(spike_nums[:, start_sce:(t + window_duration) - 1])
                        sce_tuples.append((sce_tuple[0] + index_max, sce_tuple[0] + index_max))
                        sce_bool[sce_tuple[0] + index_max] = True
                        # sce_tuples.append((start_sce, t-1))
                        sce_times_numbers[sce_tuple[0] + index_max] = len(sce_tuples) - 1
                    else:
                        # then a new SCE is detected
                        sce_bool[start_sce:(t + window_duration) - 1] = True
                        sce_tuples.append((start_sce, (t + window_duration) - 2))
                        # sce_tuples.append((start_sce, t-1))
                        sce_times_numbers[start_sce:(t + window_duration) - 1] = len(sce_tuples) - 1

                    start_sce = -1
                    cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
                if no_redundancy and cells_has_been_removed_due_to_redundancy:
                    sum_value += nb_cells_already_in_sce
                    if sum_value >= activity_threshold:
                        # then a new SCE start right after the old one
                        start_sce = t
                        cells_in_sce_so_far = np.zeros(n_cells, dtype="bool")
                        if no_redundancy:
                            # keeping only cells spiking at time t, as we're gonna shift of one on the next step
                            sum_spikes = np.sum(spike_nums[:, t])
                            pos_cells = np.where(sum_spikes)[0]
                            cells_in_sce_so_far[pos_cells] = True

    n_sces = len(sce_tuples)
    sce_nums = np.zeros((n_cells, n_sces), dtype="int16")
    for sce_index, sce_tuple in enumerate(sce_tuples):
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


def detect_sce_on_traces(raw_traces, use_speed=True, speed_threshold = None, sce_n_cells_threshold=5,
                         sce_min_distance=4, use_median_norm=True, use_bleaching_correction=False,
                         use_savitzky_golay_filt=True):

    traces = np.copy(raw_traces)

    if use_speed is True:
        if speed_threshold is None:
            speed_threshold = 1  # define below which instantaneous speed (cm/s) we consider mice in rest period
        else:
            pass
        rest_periods = np.where(speed < speed_threshold)[0]  # not used
        traces_rest = traces[:, rest_periods]  # can do detection directly on this selected part of the traces

    if use_speed is False:
        if speed_threshold is not None:
            raise Exception('speed threshold is useless if speed is not used')
        else:
            pass

    if use_median_norm is True:
        median_normalization(traces)

    if use_bleaching_correction is True:
        bleaching_correction(traces)

    if use_savitzky_golay_filt is True:
        savitzky_golay_filt(traces)

    print(f" traces normalization done with n traces: {len(traces)}")

    # Detect small transients
    n_cells, n_frames = traces.shape
    window_size = 40  # size in frames of the sliding window to detect transients
    activity_tmp_all_cells=[[] for i in range(n_cells)]

    if use_speed is True:
        print(f" starting detection using speed threshold")
        for i in range(n_cells):
            activity_tmp = np.zeros((1, n_frames))
            trace_tmp = traces[i, :]
            burst_threshold = np.median(trace_tmp) + scipy_stats.iqr(trace_tmp) / 2
            for k in range(window_size + 1, n_frames - window_size):
                 if speed [k] < speed_threshold:
                    window_tmp = np.arange(k - window_size,k + window_size)
                    median_tmp = np.median(trace_tmp[window_tmp])
                    if np.sum(activity_tmp[k - 10:k - 1]) and median_tmp < burst_threshold:
                        activity_tmp[k] = (trace_tmp[k] - median_tmp) > (3 * scipy_stats.iqr(trace_tmp[window_tmp]))
            activity_tmp_all_cells[i] = np.where(activity_tmp)[0]

    if use_speed is False:
        print(f" starting detection without speed threshold")
        for i in range(n_cells):
            activity_tmp = np.zeros(n_frames)
            trace_tmp = traces[i, :]
            burst_threshold = np.median(trace_tmp) + scipy_stats.iqr(trace_tmp) / 2
            for k in np.arange(window_size + 1, n_frames - window_size, 5):
                window_tmp = np.arange(k - window_size,k + window_size)
                median_tmp = np.median(trace_tmp[window_tmp])
                if np.sum(activity_tmp[k - 10:k - 1])==0 and median_tmp < burst_threshold:
                    activity_tmp[k] = (trace_tmp[k] - median_tmp) > (3 * scipy_stats.iqr(trace_tmp[window_tmp]))
            activity_tmp_all_cells[i] = np.where(activity_tmp)[0]
            print(f" cell #{i} has {len(activity_tmp_all_cells[i])} small activation")

    print(f" small transients detection is done")

    raster = np.zeros((n_cells, n_frames))
    for i in range(n_cells):
        raster[i, activity_tmp_all_cells[i]] = 1

    print(f" raster is obtained")

    # sum activity over 2 consecutive frames
    sum_activity=np.zeros(n_frames-1)
    for i in range( n_frames-1):
        sum_activity[i] = np.sum(np.amax(raster[:,np.arange(i,i+1)], axis = 1))

    print(f" sum activity is obtained with max is {np.max(sum_activity)}")

    #select synchronous calcium events
    sce_loc = scisi.find_peaks(sum_activity, height=sce_n_cells_threshold, distance=sce_min_distance)[0]
    n_sce = len(sce_loc)

    print(f" SCE are detected with n SCE is {n_sce}")
    print(f"sce_loc {sce_loc}")

    # create cells vs sce matrix
    sce_cells_matrix = np.zeros((n_cells, n_sce))
    for i in range(n_sce):
        sce_cells_matrix[:, i] = np.amax(raster[:, np.arange(np.max((sce_loc[i]-1, 0)),
                                                             np.min((sce_loc[i]+2, n_frames)))], axis=1)

    print(f" SCE vs cells matrix is obtained")
    return sce_cells_matrix, sce_loc

# Normalization functions
def median_normalization(traces):
    n_cells, n_frames = traces.shape
    for i in range(n_cells):
        traces[i, :] = traces[i, :] / np.median(traces[i, :])
    return traces

def bleaching_correction(traces):
    n_cells, n_frames = traces.shape
    for k in range(n_cells):
        p0 = np.polyfit(np.arange(n_frames), traces[k, :], 3)
        traces[k, :] = traces[k, :] / np.polyval(p0, np.arange(n_frames))
    return traces

def savitzky_golay_filt(traces):
    traces = scisi.savgol_filter(traces, 5, 3, axis=1)
    return traces