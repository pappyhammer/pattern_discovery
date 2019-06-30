import numpy as np
from numpy import log10, pi, exp
from neo.io import NeuralynxIO
import os
from matplotlib import pyplot as plt
import quantities as pq
from elephant.spectral import welch_psd
from datetime import datetime
from numpy.fft import fft, ifft
from ephyviewer import mkQApp, MainViewer, TraceViewer, TimeFreqViewer
from ephyviewer import InMemoryAnalogSignalSource
import ephyviewer
import pattern_discovery.tools.BOSC_analysis as bosc
from scipy import signal
import matplotlib.gridspec as gridspec
import math
from scipy.signal import butter, lfilter


def nextpow2(i):
    n = 1
    while n < i: n *= 2
    return n


def complex_morlet_wavelet(sampling_rate, frequency, s_wav=None, time_wav=None):
    # create wavelet

    # define how long is the wavelet
    if time_wav is None:
        time = np.arange(-1, 1 + 1 / sampling_rate, 1 / sampling_rate)
    else:
        time = time_wav
    # frequency in Hz
    f = frequency
    sine_wave = exp(2 * 1j * pi * f * time)

    # compute gaussian

    # n : The number of cycles of the Gaussian taper defines its width, which in turn defines the width of the
    # wavelet
    # a larger number of cycles gives you better frequency precision at the cost of worse temporal precision,
    # and a smaller number of cycles gives you better temporal precision at the cost of worse frequency precision
    if s_wav is None:
        n = 4
        s = n / (2 * pi * f)
    else:
        s = s_wav
    gaussian_win = exp(-time ** 2 / (2 * s ** 2))
    A = np.sqrt(1 / (s * np.sqrt(pi)))
    # window the sinewave by a gaussian to create complex morlet wavelet
    wavelet = A * sine_wave * gaussian_win

    # to check if the wavelet taper to zero
    # plt.plot(time, np.real(wavelet))
    # plt.show()
    return wavelet


def welch_method(data_bin_for_fft, sampling_rate, baseline_welch_power):
    # Estimate power spectral density

    # baseline_welch_power used to be patient.baseline_welch_power[channel_str]

    # using Welch’s method.
    # Welch’s method [R145] computes an estimate of the power spectral density by dividing the data into
    # overlapping segments, computing a modified periodogram for each segment
    # and averaging the periodograms.
    # parameters : window of 2s (2*sampling_rate), overlaping 50% (1s)
    # n_conv_pow2 = nextpow2(n_convolution)
    # Pxx units : V**2/Hz
    # if paramaters are changing, don't forget to change
    # the paramters in get_baseline_power function as well
    welch_freq, welch_power_spectrum = signal.welch(data_bin_for_fft, fs=sampling_rate,
                                                    window='hanning',
                                                    nperseg=sampling_rate * 2, noverlap=None,
                                                    nfft=sampling_rate * 8,  # n_conv_pow2,
                                                    detrend='constant', return_onesided=True,
                                                    scaling='density', axis=-1)
    # print(f'baseline_welch_power {np.shape(baseline_welch_power)}, '
    #       f'welch_power_spectrum {np.shape(welch_power_spectrum)}')
    # normalization :
    decibel_conversion = False
    percentage_change = True
    z_score_normalization = False
    # normalization using z-score
    if z_score_normalization:
        squared_baseline_welch_power = [x ** 2 for x in baseline_welch_power]
        welch_power_spectrum = np.subtract(welch_power_spectrum,
                                           baseline_welch_power)
        welch_power_spectrum = np.divide(welch_power_spectrum, squared_baseline_welch_power)
    elif decibel_conversion:
        # normalization by dividing by baseline
        # using decibel conversion method (see mike x cohen book page 220)
        welch_power_spectrum = 10 * log10(welch_power_spectrum /
                                          baseline_welch_power)
    elif percentage_change:
        welch_power_spectrum = np.subtract(welch_power_spectrum,
                                           baseline_welch_power)
        welch_power_spectrum = np.divide(welch_power_spectrum,
                                         baseline_welch_power)
        welch_power_spectrum = np.multiply(welch_power_spectrum, 100)
    return welch_power_spectrum, welch_freq


def apply_wavelet(sampling_rate, data_bin_for_fft, wp, baseline_power=None,
                  use_baseline=True, tenlog10=True):
    """
    Apply fft then wavelet to data_bin_for_fft
    :param wp: waveletParamaters instance
    :param data_bin_for_fft: the data to which apply fft, list of amplitudes (uV)
    :param channel_str: channel in which the wavelet is applied
    :
    :return:
    """

    # define wavelet parameters
    # time_wav define how long is the wavelet
    # If time (max-min) is too low, then if we have a min_cycle quite high, the wavelet will not taper to zero
    time_wav = np.arange(-2, 2 + 1 / sampling_rate, 1 / sampling_rate)
    if wp.log_y:
        # the peak frequencies of wavelets increase logarithmically
        frex = np.logspace(log10(wp.min_freq), log10(wp.max_freq), wp.num_frex)
    else:
        # the peak frequencies of wavelets increase linearly
        frex = np.linspace(wp.min_freq, wp.max_freq, num=wp.num_frex)
    # compute gaussian
    # the number of cycles increases from min_cycle to max_cycle in the same number of steps used
    # to increase the frequency of the wavelets from min_freq Hz to max_freq H
    if wp.increase_gaussian_cycles:
        if wp.log_y:
            s = np.logspace(log10(wp.min_cycle), log10(wp.max_cycle), wp.num_frex) / (2 * pi * frex)
        else:
            s = np.linspace(wp.min_cycle, wp.max_cycle, wp.num_frex) / (2 * pi * frex)
    else:
        s = wp.wav_cycles / (2 * pi * frex)
    # define convolution parameters
    # print(f'len wav {len(time_wav)}')
    n_wavelet = len(time_wav)
    n_data = len(data_bin_for_fft)  # data_last_index_range - data_first_index_range
    n_convolution = n_wavelet + n_data - 1
    n_conv_pow2 = nextpow2(n_convolution)
    half_of_wavelet_size = (n_wavelet - 1) // 2

    # print(f'max data_bin_for_fft: {np.max(data_bin_for_fft)}, min data_bin_for_fft: {np.min(data_bin_for_fft)} ')
    # get FFT of data
    eegfft = fft(data_bin_for_fft, n_conv_pow2)

    # initialize
    temppower = np.zeros([wp.num_frex, n_data])
    for fi in range(wp.num_frex):
        # complex morlet wavelet
        wavelet = complex_morlet_wavelet(sampling_rate, frex[fi], s[fi], time_wav)
        # wavelet_fft = fft(np.sqrt(1 / (s[fi] * np.sqrt(pi))) * exp(2 * 1j * pi * frex[fi] * time_wav) * exp(
        #     -time_wav ** 2 / (2 * (s[fi] ** 2))), n_conv_pow2)
        # take the fft of the wavelet
        wavelet_fft = fft(wavelet, n_conv_pow2)

        # convolution
        eegconv = ifft(wavelet_fft * eegfft, n_conv_pow2)  # convolution theorem
        eegconv = eegconv[:n_convolution]
        eegconv = eegconv[half_of_wavelet_size:-half_of_wavelet_size]

        # print(f'max eegconv: {np.max(eegconv)}, min eegconv: {np.min(eegconv)} ')
        # average power over trials
        # this performs baseline transform, which is covered in more depth in ch18
        # decibel- normalized power
        # has been modified comparing to the book
        temppower[fi, :] = np.absolute(eegconv) ** 2

        # print(f'0 : {baseidx[0]}, 1 : {baseidx[1]},  len {len(temppower[baseidx[0]:baseidx[1]])}')
        # eegpower[fi, :] = 10 * log10(temppower / np.mean(temppower[:int(0.3*sampling_rate)]))

    # The baseline is a period of time, typically a few hundred milliseconds before the start of the trial,
    # when little or no task-related processing is expected. The choice of baseline period is a nontrivial one
    # and influences the interpretation of your results. There are several important issues to consider when
    # selecting a baseline period, which are discussed later in this chapter (section 18.10).
    if (wp.baseline_mode is not None) and use_baseline:
        if baseline_power is None:
            print("apply_wavelet: baseline_powerbaseline_power is None")

    if (wp.baseline_mode is not None) and use_baseline and (baseline_power is not None):
        # normlization using z-score
        if wp.baseline_mode == "z_score_normalization":
            mean_baseline_power = np.mean(baseline_power, axis=1)
            mean_baseline_power = np.reshape(mean_baseline_power, [len(mean_baseline_power), 1])
            numerator = np.subtract(temppower, mean_baseline_power)

            denominator = np.reshape(np.std(baseline_power, axis=1), [len(baseline_power), 1])

            dbconverted = np.divide(numerator, denominator)
        elif wp.baseline_mode == "decibel_conversion":
            # print("decibel_conversion is on")
            # normalization by dividing by baseline
            # using decibel conversion method (see mike x cohen book page 220)
            mean_baseline_power = np.mean(baseline_power, axis=1)
            # print(f'max mean_baseline_power: {np.max(mean_baseline_power)}, '
            #       f'min mean_baseline_power: {np.min(mean_baseline_power)} ')
            # print(f'max temppower: {np.max(temppower)}, '
            #       f'min temppower: {np.min(temppower)} ')
            mean_baseline_power = np.reshape(mean_baseline_power, [len(mean_baseline_power), 1])
            dbconverted = 10 * log10(temppower / mean_baseline_power)
        elif wp.baseline_mode == "percentage_change":
            mean_baseline_power = np.mean(baseline_power, axis=1)
            mean_baseline_power = np.reshape(mean_baseline_power, [len(mean_baseline_power), 1])
            dbconverted = np.subtract(temppower, mean_baseline_power)
            dbconverted = np.divide(dbconverted, mean_baseline_power)
            dbconverted = np.multiply(dbconverted, 100)
        else:
            dbconverted = 10 * log10(temppower)
    elif tenlog10:
        dbconverted = 10 * log10(temppower)
    else:
        dbconverted = temppower

    # print(f'max dbconverted: {np.max(dbconverted)}, min dbconverted: {np.min(dbconverted)} ')
    # # normalization
    # 1 == std, and 0 = the mean, but not as robust as median
    # dbconverted = (dbconverted - np.mean(dbconverted)) / np.std(dbconverted)
    # then median = 1 on y ordinate
    # normalization if done later
    # dbconverted = dbconverted / np.median(dbconverted)
    return dbconverted, frex, n_convolution


def apply_wavelet_to_time_segment(data_bin, sampling_rate, wp, beg_time, end_time,
                                  tenlog10=True,
                                  baseline_welch_power=None):
    """

    :param data_bin: data from LFP, data_bin is not an appropriate way to name it.
    :param channel_str:
    :param wp:
    :param beg_time: correspond to an index of data_bin (from patient.EEG_info)
    :param end_time:
    :param baseline_welch_power: needed if wp.welch_method is True
    :return:
    """
    eeg_times = np.arange(beg_time, end_time)
    data_bin = np.array(data_bin)
    # adding +1 because diff will remove 1
    # will crash if end_time == len(data_bin) -1
    data_bin_for_fft_diff = data_bin[beg_time:
                                     end_time + 1]
    data_bin_for_fft = data_bin[beg_time:
                                end_time]

    # doing the diff to equalize the powers (derivative)
    if wp.use_derivative or wp.bosc_method:
        data_bin_for_fft_diff = np.diff(data_bin_for_fft_diff)
    else:
        data_bin_for_fft_diff = data_bin_for_fft
    # data_bin_for_fft_diff = data_bin_for_fft

    if wp.bosc_method:
        dbconverted, frex, n_convolution = bosc.bosc_tf(sampling_rate=sampling_rate,
                                                        data_bin_for_fft=data_bin_for_fft_diff,
                                                        wp=wp)
    else:
        dbconverted, frex, n_convolution = apply_wavelet(sampling_rate=sampling_rate,
                                                         data_bin_for_fft=data_bin_for_fft_diff,
                                                         wp=wp, tenlog10=tenlog10,
                                                         use_baseline=False)
        print(f'in apply_wavelet_to_time_segment: apply_wavelet is done')
        # dbconverted contains the powers (float numbers), not the phase,

    # Estimate power spectral density

    # using Welch’s method.
    # Welch’s method [R145] computes an estimate of the power spectral density by dividing the data into
    # overlapping segments, computing a modified periodogram for each segment
    # and averaging the periodograms.
    if wp.welch_method:
        power_spectrum, power_spect_freq = welch_method(data_bin_for_fft=data_bin_for_fft_diff,
                                                        baseline_welch_power=baseline_welch_power)
    else:
        # data is already normalized with baseline
        power_spectrum = np.mean(dbconverted, axis=1)
        power_spectrum = (np.reshape(power_spectrum, [len(power_spectrum), 1])).flatten()
        power_spect_freq = frex

    max_value_dbconverted = np.max(dbconverted)

    # if compute_plots or wp.compute_freq_bands or use_spike_removal or compute_correlation_matrices:
    #     db_to_pass = dbconverted
    # else:
    #     # to save memory
    #     db_to_pass = None
    #
    # game_events_time_stamp_hz = [game.events[e].time_stamp_hz for e
    #                              in objects_indexes]
    # if is_retrieving:
    #     title = f'{patient.name} ' \
    #             f'retrieval of {len(objects_indexes)}  objects, channel {channel_str} '
    # else:
    #     title = f'{patient.name} ' \
    #             f'encoding {len(objects_indexes)}  objects, channel {channel_str} '

    return WaveletOutcome(eeg_times, frex, data_bin_for_fft,
                          power_spect_freq=power_spect_freq, power_spectrum=power_spectrum,
                          sampling_rate=sampling_rate,
                          max_value_dbconverted=max_value_dbconverted), dbconverted


class FrequencyBandEpisode:
    """
    Represent a frequency band episode
    """

    def __init__(self, freq_limits, duration, starting_point, sampling_rate, **kwargs):
        """

        :param freq_limits: list of 2 integers
        :param duration: time duration
        :param starting_point: time point (from the start of the ampli)
        :param freq_indexes : set of freq_index and time_index
        """
        self.freq_limits = freq_limits
        self.duration = duration  # in Hz
        self.starting_point = starting_point
        self.end_point = starting_point + duration
        self.sampling_rate = sampling_rate

    def __str__(self):
        return f'{self.freq_limits[0]} {self.freq_limits[1]} {self.duration} {self.starting_point} {self.sampling_rate}'


class FrequencyBandEpisodeFirstVersion(FrequencyBandEpisode):
    """
    Represent a frequency band episode
    """

    def __init__(self, freq_limits, duration, starting_point, sampling_rate, **kwargs):
        """

        :param freq_limits: list of 2 integers
        :param duration:
        :param starting_point:
        :param freq_indexes : set of freq_index and time_index
        :param wav_outcome WaveletOutcome object
        """
        FrequencyBandEpisode.__init__(self, freq_limits, duration, starting_point, sampling_rate)
        # self.freq_limits = freq_limits
        # self.duration = duration  # in Hz
        # self.starting_point = starting_point
        # self.sampling_rate = sampling_rate

        # wav_outcome=None, freq_indexes=(0, 0)
        # not saved on file, used for computing other freq_band_ep
        if kwargs is not None:
            if "wav_outcome" in kwargs:
                self.wav_outcome = kwargs["wav_outcome"]
            else:
                self.wav_outcome = None
            if "freq_indexes" in kwargs:
                freq_indexes = kwargs["freq_indexes"]
                self.coords = set()
                self.coords.add(freq_indexes)
                self.min_freq_index = freq_indexes[0]
                self.max_freq_index = freq_indexes[0]
                self.min_time_index = freq_indexes[1]
                self.max_time_index = freq_indexes[1]
            else:
                self.coords = set()
                self.coords.add((0, 0))
                self.min_freq_index = 0
                self.max_freq_index = 0
                self.min_time_index = 0
                self.max_time_index = 0
        else:
            self.wav_outcome = None
            self.coords = set()
            self.coords.add((0, 0))
            self.min_freq_index = 0
            self.max_freq_index = 0
            self.min_time_index = 0
            self.max_time_index = 0
        # dict use to keep the min and max time index for a given freq on this freq_band
        self.min_time_index_for_freq_index = {}
        self.max_time_index_for_freq_index = {}
        # self.grow_num=0

    # def __str__(self):
    #     return f'{self.freq_limits[0]} {self.freq_limits[1]} {self.duration} {self.starting_point} {self.sampling_rate}'

    def mean_freq(self):
        """
        Return the mean frequency of the band (mean betwen the min and the max freq)
        :return:
        """
        if self.wav_outcome is None:
            return self.freq_limits[0] + ((self.freq_limits[1] - self.freq_limits[0]) / 2)

        if self.min_freq_index == self.max_freq_index:
            if self.min_freq_index == 0:  # the simple_way method has been used so to find frequency band
                return self.freq_limits[0] + ((self.freq_limits[1] - self.freq_limits[0]) / 2)
            else:
                return self.wav_outcome.frex[self.min_freq_index]
        else:
            return self.wav_outcome.frex[self.min_freq_index] + \
                   ((self.wav_outcome.frex[self.max_freq_index] - self.wav_outcome.frex[self.min_freq_index]) / 2)

    def add_coord(self, coord):
        """
        Add this coord (freq_index, time_index) to coords list, plus set the min and max for freq and time index
        :param coord:
        :return:
        """

        self.coords.add(coord)

        change = False
        if coord[0] < self.min_freq_index:
            self.min_freq_index = coord[0]
            # change = True
        elif coord[0] > self.max_freq_index:
            self.max_freq_index = coord[0]
            # change = True
        if coord[1] < self.min_time_index:
            self.min_time_index = coord[1]
            self.starting_point = self.wav_outcome.eeg_times[self.min_time_index]
            change = True
        elif coord[1] > self.max_time_index:
            self.max_time_index = coord[1]
            change = True

        if coord[0] in self.min_time_index_for_freq_index:
            if self.min_time_index_for_freq_index[coord[0]] > coord[1]:
                self.min_time_index_for_freq_index[coord[0]] = coord[1]
        else:
            self.min_time_index_for_freq_index[coord[0]] = coord[1]

        if coord[0] in self.max_time_index_for_freq_index:
            if self.max_time_index_for_freq_index[coord[0]] < coord[1]:
                self.max_time_index_for_freq_index[coord[0]] = coord[1]
        else:
            self.max_time_index_for_freq_index[coord[0]] = coord[1]

        if change:
            self.duration = self.wav_outcome.eeg_times[self.max_time_index] - \
                            self.wav_outcome.eeg_times[self.min_time_index]

    # TODO : doesn't work yet
    def grow(self, dbconverted_boolean, start, min_freq, max_freq):
        """
        Look to the 8 "cases" around if some are True, use a recursive call to continue looking.
        When a case is added, then the value is put to False
        Going from left to right, and bottom to top
        :param dbconverted_boolean
        :param start: tuple of freq_index and time_index
        :param min_freq
        :param max_freq
        :return:
        """
        # self.grow_num += 1
        # print(f'num : {self.grow_num}, coord : {start}')
        # print(f'before grow len(self.coords) {len(self.coords)}')
        set_coord_to_explore = set()
        set_coord_to_explore.add(start)
        # print(f'max_freq {max_freq}, self.wav_outcome.frex[max_freq]: {self.wav_outcome.frex[max_freq]}')
        while len(set_coord_to_explore) > 0:
            # remove one element
            coord = set_coord_to_explore.pop()
            if dbconverted_boolean[coord[0], coord[1]]:
                # going horizontal
                times_to_explore = np.arange(coord[1], len(self.wav_outcome.eeg_times[coord[1]:]))
                if len(times_to_explore) == 0:
                    times_to_explore = [coord[1]]
                for index_t, t in enumerate(times_to_explore):
                    # checking if the next one is True, if so adding it to this freq_band
                    if dbconverted_boolean[coord[0], t]:
                        dbconverted_boolean[coord[0], t] = False
                        self.add_coord((coord[0], t))
                        if ((coord[0] + 1) < len(dbconverted_boolean[:, t])) & ((coord[0] + 1) <= max_freq):
                            # add it only if not already in the list to explore
                            set_coord_to_explore.add((coord[0] + 1, t))
                            if (t + 1) < len(dbconverted_boolean[coord[0], :]):
                                # add it only if not already in the list to explore
                                set_coord_to_explore.add((coord[0] + 1, t + 1))
                            if t > 0:
                                set_coord_to_explore.add((coord[0] + 1, t - 1))
                        if coord[0] - 1 >= min_freq:
                            set_coord_to_explore.add((coord[0] - 1, t))
                            if (t + 1) < len(dbconverted_boolean[coord[0], :]):
                                # add it only if not already in the list to explore
                                set_coord_to_explore.add((coord[0] - 1, t + 1))
                            if t > 0:
                                set_coord_to_explore.add((coord[0] - 1, t - 1))
                    else:
                        break

        # print(f'after grow len(self.coords) {len(self.coords)}')
        return dbconverted_boolean


# TODO to optimize : insert at the right place (asecending starting point time) and keep a disctionnary
def insert_freq_band_if_not_overlapping(freq_band_to_insert, freq_bands, overlap_perc, hz_gap):
    """
    Insert freq_band_to_insert in freq_bands if no overlapping (as overlap_perc of the other band) from
    than hz_gap hz between 2 bands. If freq_band_to_insert is bigger than one already present, remove the one present
    and insert this one
    :param freq_band_to_insert:
    :param freq_bands:
    :param overlap_perc: ex : 0,7 = 70% overlapping
    :param hz_gap: gap in Hz (if 2 band overlap but gap > hz_gap, then both are kept)
    :return: a new freq_bands
    """
    if len(freq_bands) == 0:
        freq_bands.append(freq_band_to_insert)
        return freq_bands

    f_i = 0
    while True:
        i_s_p = freq_band_to_insert.starting_point
        i_duration = freq_band_to_insert.duration
        # i end point
        i_e_p = i_s_p + i_duration
        i_freq = freq_band_to_insert.mean_freq()

        f_i_s_p = freq_bands[f_i].starting_point
        f_i_duration = freq_bands[f_i].duration
        # f_i end point
        f_i_e_p = f_i_s_p + f_i_duration
        f_i_freq = freq_bands[f_i].mean_freq()

        # if the band episode we want to insert if starting after the end of the band in the list, then we insert it
        # it saves some time, no need to compare with the next one

        if f_i_e_p < i_s_p:
            freq_bands.insert(f_i + 1, freq_band_to_insert)
            return freq_bands

        if ((i_s_p <= f_i_s_p <= i_e_p) &
                (i_s_p <= f_i_e_p <= i_e_p) &
                (np.abs(i_freq - f_i_freq) < hz_gap)):
            del freq_bands[f_i]
        # freq_band_to_insert is contained in another band
        elif ((f_i_s_p <= i_s_p <= f_i_e_p) &
              (f_i_s_p <= i_e_p <= f_i_e_p) &
              (np.abs(i_freq - f_i_freq) < hz_gap)):
            return freq_bands
        # if the band i is starting inside of the band f_i in the time domain,
        # and the band i  > 70% in the band f_i (in time domain),
        # and there are less than 2 HZ, then we remove i
        elif ((f_i_s_p <= i_s_p <= f_i_e_p) &
              ((f_i_e_p - i_s_p) > (overlap_perc * i_duration)) &
              (np.abs(i_freq - f_i_freq) < hz_gap)):
            return freq_bands
        elif ((i_s_p <= f_i_s_p <= i_e_p) &
              ((i_e_p - f_i_s_p) > (overlap_perc * f_i_duration)) &
              (np.abs(i_freq - f_i_freq) < hz_gap)):
            del freq_bands[f_i]
        elif ((f_i_s_p <= i_e_p <= f_i_e_p) &
              ((i_e_p - f_i_s_p) > (overlap_perc * i_duration)) &
              (np.abs(i_freq - f_i_freq) < hz_gap)):
            return freq_bands
        elif ((i_s_p <= f_i_e_p <= i_e_p) &
              ((f_i_e_p - i_s_p) > (overlap_perc * f_i_duration)) &
              (np.abs(i_freq - f_i_freq) < hz_gap)):
            del freq_bands[f_i]
        else:
            f_i += 1
        if f_i >= len(freq_bands):
            break

    freq_bands.append(freq_band_to_insert)
    return freq_bands


def find_frequency_band_episodes(wav_outcome, freq_band, min_duration, threshold, hz_gap, step, simple_way=False,
                                 freq_steps=None):
    """

    :param wav_outcome: a WaveletOutcome object
    :param freq_band: list of two integers, representing the min and max of the frequency band
    :param min_duration: min duraiton of an episode, in s
    :param threshold: threshold to select frequencies episodes
    :param simple_way:
    :param freq_steps : step (in Hz) between each frequency for wavelet computation, not valid if log scale
    :param step : step that will be applied to search for frequency band. if (1 / freq_steps),
    step will be of 1, except if the freq_steps is more than 1Hz, then it will bug
    if step = 1 will jump from 1 frequency to the other one depending on the frequency step used by the wavelet
    :param hz_gap: max gap necessery between 2 overlap episode to be counted as one
    :return: list of FrequencyBandEpisode objects
    """
    # going trhough time
    episode_start = None
    freq_bands = []
    # corresponding frequencies in the frex array (getting the index to apply to dbconverted)
    freq_min_index = np.searchsorted(wav_outcome.frex, freq_band[0], side="left")
    freq_max_index = np.searchsorted(wav_outcome.frex, freq_band[1], side="left")
    freq_indices = np.arange(freq_min_index, freq_max_index, 1)
    duration_hz = min_duration * wav_outcome.sampling_rate

    # print(f'np.min(wav_outcome.eeg_times) : {np.min(wav_outcome.eeg_times/wav_outcome.sampling_rate)} '
    #       f'{np.max(wav_outcome.eeg_times/wav_outcome.sampling_rate)}')

    # using simple way, separing the frequency band in a few frequency band of 1Hz
    if simple_way:
        # variable used to filter the bands
        overlap_perc = 0.65
        # making a list of frequency range, as tuple
        # TODO : going from low range freq to high range by step of 0.1 Hz with band of 1 Hz
        list_frex_band_index = []
        # frex_range = np.linspace(freq_band[0], freq_band[1], (freq_band[1]-freq_band[0])*2)
        # i = 0
        first_index_frex = np.searchsorted(wav_outcome.frex, freq_band[0], side="left")
        last_index_frex = np.searchsorted(wav_outcome.frex, freq_band[1], side="right")
        i = first_index_frex
        # step use to choose frequency_bands through which we will select episode band
        # if step = 1 will jump from 1 frequency to the other one depending on the frequency step used by the wavelet
        # step = 1
        #
        while i < (last_index_frex - step):
            list_frex_band_index.append((i, i + step))
            i += step

        for freq_band_index in list_frex_band_index:
            episode_start = None
            for e in np.arange(len(wav_outcome.eeg_times)):
                # if true at least one power is superior to threshold
                if np.any(wav_outcome.dbconverted[freq_band_index[0]:freq_band_index[1], e] > threshold):
                    if episode_start is None:
                        episode_start = wav_outcome.eeg_times[e]
                else:
                    if episode_start is not None:
                        # testing if the duration of the episode is superior to the duration indicated
                        if (wav_outcome.eeg_times[e] - episode_start) > duration_hz:
                            # print(f'e : {e}, time : {episode_start / wav_outcome.sampling_rate}')
                            freq_b = FrequencyBandEpisodeFirstVersion((wav_outcome.frex[freq_band_index[0]],
                                                                       wav_outcome.frex[freq_band_index[1]]),
                                                                      duration=(
                                                                              wav_outcome.eeg_times[e] - episode_start),
                                                                      starting_point=episode_start,
                                                                      sampling_rate=wav_outcome.sampling_rate,
                                                                      wav_outcome=wav_outcome)
                            freq_bands = insert_freq_band_if_not_overlapping(freq_b, freq_bands, overlap_perc, hz_gap)
                        episode_start = None
            if episode_start is not None:
                if (wav_outcome.eeg_times[-1] - episode_start) > duration_hz:
                    freq_b = FrequencyBandEpisodeFirstVersion((wav_outcome.frex[freq_band_index[0]],
                                                               wav_outcome.frex[freq_band_index[1]]),
                                                              duration=(wav_outcome.eeg_times[-1] - episode_start),
                                                              starting_point=episode_start,
                                                              sampling_rate=wav_outcome.sampling_rate,
                                                              wav_outcome=wav_outcome)
                    freq_bands = insert_freq_band_if_not_overlapping(freq_b, freq_bands, overlap_perc, hz_gap)
        return freq_bands
    # doesn't work yet
    else:
        freq_bands = []
        dbconverted_boolean = wav_outcome.dbconverted > threshold
        for e in np.arange(len(wav_outcome.eeg_times)):
            for f in freq_indices:
                # print('for f in freq_indices')
                if dbconverted_boolean[f, e]:
                    # print(f'dbconverted_bool: {dbconverted_boolean[f:f+2,e:e+30]}')
                    # print('dbconverted is True')
                    # mean that this power cell is not part of any freq_band saved
                    # as it is set to False when added
                    new_freq_band = FrequencyBandEpisodeFirstVersion(freq_band, starting_point=wav_outcome.eeg_times[e],
                                                                     duration=1,
                                                                     sampling_rate=wav_outcome.sampling_rate,
                                                                     wav_outcome=wav_outcome,
                                                                     coord=(f, e))
                    dbconverted_boolean = new_freq_band.grow(dbconverted_boolean=dbconverted_boolean, start=(f, e),
                                                             min_freq=freq_min_index, max_freq=freq_max_index)
                    # Adding freq band whose duration is superior than the min_duration
                    if new_freq_band.duration >= duration_hz:
                        freq_bands.append(new_freq_band)
                    # test, should be False then
                    if dbconverted_boolean[f, e]:
                        print('dbconverted_boolean still True')
        return freq_bands


def spike_detector(wav_time_freq, wp, sampling_rate, frex, eeg_times, data_bin_for_fft,
                   threshold, events_time_stamp_hz=None):
    # no more spike removal, it's done when loading spikes file
    remove_spike = False
    if threshold is None:
        if wp.using_median_for_threshold:
            threshold = np.median(wav_time_freq)
        else:
            threshold = np.mean(wav_time_freq) + np.std(wav_time_freq)
    dbconverted_boolean = wav_time_freq > (threshold * wp.spike_threshold_ratio)
    index_low_Hz = np.searchsorted(frex, wp.low_freq_spike_detector, side="left")
    # index_mid_Hz = np.searchsorted(frex, 8, side="left")
    index_top_Hz = np.searchsorted(frex, wp.high_freq_spike_detector, side="left")
    # num_frex = len(frex)
    len_times = len(eeg_times)
    spikes_times = []
    # keep times in Hz of the spikes, without correction to anticipate the fact we're gonna erase the previous one,
    # changing the time of the following ones
    spikes_original_times = []
    len_pow_spike = int(sampling_rate / 100)
    e = 0
    print(f'n Hz {index_top_Hz-index_low_Hz}')
    print(f"threshold {threshold}")
    print(f"spike_threshold_ratio {wp.spike_threshold_ratio}")
    print(f"threshold*wp.spike_threshold_ratio {threshold*wp.spike_threshold_ratio}")
    # remove sampling_rate due to wavelet border effect
    while e < (len_times - sampling_rate):
        # count_low = np.count_nonzero(dbconverted_boolean[index_low_Hz:index_mid_Hz, e:(e + len_pow_spike)])
        # count_top = np.count_nonzero(dbconverted_boolean[index_mid_Hz:index_top_Hz, e:(e + len_pow_spike)])
        total_count = np.count_nonzero(dbconverted_boolean[index_low_Hz:index_top_Hz, e:(e + len_pow_spike)])
        # if (count_low > ((percentage_threshold * (index_mid_Hz - index_low_Hz)) * len_pow_spike)) or \
        #         (count_top > ((percentage_threshold * (index_top_Hz - index_mid_Hz)) * len_pow_spike)):
        total_count = np.sum(np.any(dbconverted_boolean[index_low_Hz:index_top_Hz, e:(e + len_pow_spike)], axis=1))

        # if total_count > ((wp.spike_percentage * (index_top_Hz - index_low_Hz)) * len_pow_spike):
        if total_count > (wp.spike_percentage * (index_top_Hz - index_low_Hz)):
            print(f"total_count {e / sampling_rate} {total_count} / {index_top_Hz - index_low_Hz} : "
                  f"{(total_count / (index_top_Hz - index_low_Hz)) * 100}%")
            spikes_times.append(eeg_times[e] + (len_pow_spike // 2))
            spikes_original_times.append(eeg_times[e] + (len_pow_spike // 2))
            e += int(sampling_rate / 5)
            # e += 1
        else:
            # To go faster, if there is no power > threshold in less than 20% of the sampling_rate/4 area
            # then jumping directly to the next segment
            # if (count_low < ((0.2 * (index_mid_Hz - index_low_Hz)) * len_pow_spike)) and \
            #         (count_top < ((0.2 * (index_top_Hz - index_mid_Hz)) * len_pow_spike)):
            # if total_count > ((0.2 * (index_top_Hz - index_low_Hz)) * len_pow_spike):
            #     e += len_pow_spike
            # else:
            #     # Not too jump one by one, for 1024 sr it would mean 10 by 10
            #     e += int(sampling_rate / 100)
            e += int(sampling_rate / 500)

    return spikes_times, spikes_original_times  # , eeg_times, wav_time_freq, data_bin_for_fft


def bosc_frequency_band_episodes_finder(wav_outcome, wp, fbi, powthresh, durthresh):
    # find_frequency_band_episodes(wav_outcome=wav_outcome,
    freq_band = [max(fbi.low_freq, wp.min_freq),
                 min(fbi.high_freq, wp.max_freq)]
    # freq_steps = wp.freq_steps
    # hz_gap = wp.hz_gap_fb_ep
    # step = wp.step_fb_ep
    p_episode_sum = 0
    freq_band_episodes = []
    p_episodes_dict = {}

    freq_min_index = np.searchsorted(wav_outcome.frex, freq_band[0], side="left")
    freq_max_index = np.searchsorted(wav_outcome.frex, freq_band[1], side="left")
    freq_indices = np.arange(freq_min_index, freq_max_index, 1)

    for f in freq_indices:
        detected, h = bosc.bosc_detect(wav_outcome.dbconverted[f, :], powthresh[f], durthresh[f])
        p_episode = len(np.flatnonzero(detected)) / len(detected)
        p_episodes_dict[wav_outcome.frex[f]] = p_episode
        p_episode_sum += p_episode
        # print(f'len(np.flatnonzero(detected)) {len(np.flatnonzero(detected))}')
        # print(f'p_episode: {np.round(p_episode, 3)}')
        if detected.any():
            for i in np.arange(0, h[0, :].size):
                fbe = FrequencyBandEpisodeBosc(freq_limits=freq_band, duration=h[1, i] - h[0, i],
                                               starting_point=wav_outcome.eeg_times[h[0, i]],
                                               sampling_rate=wav_outcome.sampling_rate,
                                               freq=wav_outcome.frex[f])
                freq_band_episodes.append(fbe)
                # freq_bands.append((h[0, i], h[1, i]))

    mean_p_episode = p_episode_sum / len(freq_indices)
    return freq_band_episodes, p_episodes_dict, mean_p_episode


def plot_correlation_matrix(wav_outcome, wp, title, save_file_name=None, show_plot=False):
    fig = plt.figure(figsize=[16, 12], tight_layout=True)
    fig.canvas.set_window_title(title)
    # plt.title(title, fontsize=6)
    ax = fig.add_subplot(111)
    ax.set_title(title, fontsize=5)
    freq_min_index = np.searchsorted(wav_outcome.frex, wp.min_freq, side="left")
    freq_max_index = np.searchsorted(wav_outcome.frex, wp.max_freq, side="left")
    dbconverted = wav_outcome.dbconverted[freq_min_index:freq_max_index, :]
    transpose_mat = dbconverted.transpose()
    df = pd.DataFrame(transpose_mat)
    corr = df.corr()
    # current_palette = sns.color_palette()
    step = int(5 / wp.freq_steps)
    x_ticks_lab = wav_outcome.frex[freq_min_index:freq_max_index:step]
    x_ticks_lab = np.floor(x_ticks_lab)
    # print(f'x_ticks_lab {x_ticks_lab}, len {len(x_ticks_lab)}, old {wav_outcome.frex[freq_min_index:freq_max_index]}')
    y_ticks_lab = x_ticks_lab
    ax_sns = sns.heatmap(corr, cmap=plt.cm.jet, xticklabels=x_ticks_lab, yticklabels=y_ticks_lab)
    ax_sns.set_xticks(np.arange(0, freq_max_index - freq_min_index, step))
    ax_sns.invert_yaxis()
    ax_sns.set_yticks(np.arange(0, freq_max_index - freq_min_index, step))
    # plt.matshow(corr)
    # plt.colorbar()
    plt.title(title)

    if save_file_name is not None:
        fig.savefig(save_file_name)

    if show_plot:
        plt.show()
        plt.close(fig)

    return ''

def spectral_analysis_on_time_segment(beg_time, lfp_signal, sampling_rate, n_times, window_len_in_s,
                      save_formats, wavelet_param, file_name, param, dpi=400,
                      save_spectrogram=True, keep_dbconverted=False):

    window_len_in_times = window_len_in_s * sampling_rate
    beg_time_s = beg_time // sampling_rate
    # for ploting : represents the automatically-chosen levels that will be contour up to
    num_levels_contourf = 50
    wp = wavelet_param
    wav_outcome, dbconverted = apply_wavelet_to_time_segment(data_bin=lfp_signal,
                                                             sampling_rate=sampling_rate,
                                                             wp=wp,
                                                             beg_time=beg_time,
                                                             end_time=beg_time +
                                                                      int(window_len_in_s * sampling_rate))
    # TODO: see if always necessary, use a lot of memory
    wav_outcome.dbconverted = dbconverted
    if wp.using_median_for_threshold:
        threshold = np.median(dbconverted)
    else:
        threshold = np.mean(dbconverted) + np.std(dbconverted)

    if wp.detect_spikes:
        print(f'Spike detector start for distance')
        # doing it before computing threshold, because we're using a treshold computed from the
        # wavelet outcome itself
        wav_outcome.spikes, wav_outcome.original_spikes = \
            spike_detector(wav_time_freq=wav_outcome.dbconverted, wp=wp,
                           sampling_rate=wav_outcome.sampling_rate,
                           frex=wav_outcome.frex,
                           eeg_times=wav_outcome.eeg_times,
                           data_bin_for_fft=wav_outcome.data_bin_for_fft,
                           threshold=None,
                           events_time_stamp_hz=wav_outcome.events_time_stamp_hz)

        print(f'Nb of spikes in the segment '
              f'{beg_time_s} sec to {beg_time_s + window_len_in_s} sec : '
              f'{len(wav_outcome.spikes)}')
        # TODO: keep in recorded_animal all spikes detected

    # use to set to limit of the colorbar for the heatmap
    max_color_heatmap = np.max(wav_outcome.dbconverted)

    if wp.compute_freq_bands and not wp.bosc_method:
        # print(
        #     f'Apply through wav, freq band finding '
        #     f'for {self.recorded_animal.animal_id}_{self.day_str}_{self.location}_{self.segment_id}')
        # fbi : Game.FREQ_BAND_ID objects
        # wav_outcome.all_freq_bands = []
        for fbi_str in wp.freq_band_to_explore:
            if fbi_str in TYPE_OF_FREQ_BAND:
                fbi = TYPE_OF_FREQ_BAND[fbi_str]
            else:
                continue
            if fbi.high_freq <= wp.min_freq:
                continue
            if fbi.low_freq >= wp.max_freq:
                continue
            # print(f'{fbi.name} start')
            freq_band_ep_list = find_frequency_band_episodes(wav_outcome=wav_outcome,
                                                             freq_band=[max(fbi.low_freq, wp.min_freq),
                                                                        min(fbi.high_freq,
                                                                            wp.max_freq)],
                                                             min_duration=fbi.width_detection,
                                                             threshold=threshold,
                                                             simple_way=True,
                                                             freq_steps=wp.freq_steps,
                                                             hz_gap=wp.hz_gap_fb_ep,
                                                             step=wp.step_fb_ep)
            wav_outcome.all_freq_bands.extend(freq_band_ep_list)
            # game.freq_bands[channel_str][f'{fbi.var_name}_{action_str}'] = freq_band_ep_list
            # print(f'{fbi.name} end')
            """
            save_freq_band_ep
            freq_bands_episodes, p_episodes_dict, mean_p_episode = \
                        bosc_frequency_band_episodes_finder(wav_outcome, wp, fbi, powthresh, durthresh)
            stats are done in stat_and_outcomes in data_analysis
            """
    elif wp.bosc_method:
        pv, meanpower = bosc.bosc_bgfit(wav_outcome.frex, dbconverted)

        powthresh, durthresh = bosc.bosc_thresholds(sampling_rate, 0.95, 3, wav_outcome.frex, meanpower)

        for fbi_str in wp.freq_band_to_explore:
            if fbi_str in TYPE_OF_FREQ_BAND:
                fbi = TYPE_OF_FREQ_BAND[fbi_str]
            else:
                continue
            if fbi.high_freq <= wp.min_freq:
                continue
            if fbi.low_freq >= wp.max_freq:
                continue
            print(f'{fbi.name} start bosc')
            freq_bands_episodes, p_episodes_dict, mean_p_episode = \
                bosc_frequency_band_episodes_finder(wav_outcome, wp, fbi, powthresh, durthresh)
            # print(f'{self.recorded_animal.animal_id}_{self.day_str}_{self.location}_{self.segment_id}, '
            #       f'{beg_time_s} sec to {beg_time_s + window_len_in_s} sec : '
            #       f'freq_band: {fbi_str}, mean Pepisode {np.round(mean_p_episode, 3)}')
            wav_outcome.all_freq_bands.extend(freq_bands_episodes)
            # TODO: save the result of p_episode computation
            # if channel_str not in game.freq_bands:
            #     game.freq_bands[channel_str] = dict()
            # game.freq_bands[channel_str][f'{fbi.var_name}_{action_str}'] = freq_bands_episodes
            # if channel_str not in game.mean_p_episode:
            #     game.mean_p_episode[channel_str] = dict()
            # game.mean_p_episode[channel_str][f'{fbi.var_name}_{action_str}'] = mean_p_episode
    if save_spectrogram:
        plot_wavelet_heatmap(threshold=threshold,
                             num_levels_contourf=num_levels_contourf,
                             log_y=wp.log_y,
                             display_EEG=True,
                             sampling_rate=sampling_rate,
                             wp=wp,
                             wav_outcome=wav_outcome, max_v_colorbar=max_color_heatmap,
                             param=param,
                             file_name=file_name + f"_{beg_time_s}s_win_{window_len_in_s}",
                             plot_freq_band_detection=wp.show_freq_bands,
                             dpi=dpi,
                             levels=None, red_v_bars=None, save_formats=save_formats)
    if not keep_dbconverted:
        wav_outcome.dbconverted = None
    return wav_outcome


def spectral_analysis(lfp_signal, sampling_rate, n_times, window_len_in_s,
                      save_formats, wavelet_param, file_name, param, dpi=400,
                      save_spectrogram=True, keep_dbconverted=False):
    wav_outcomes = []
    window_len_in_times = window_len_in_s * sampling_rate
    for time_index, beg_time in enumerate(np.arange(0, n_times, window_len_in_times)):
        wav_outcome = spectral_analysis_on_time_segment(beg_time, lfp_signal, sampling_rate, n_times, window_len_in_s,
                                                        save_formats, wavelet_param, file_name, param, dpi,
                                                        save_spectrogram, keep_dbconverted)
        wav_outcomes.append(wav_outcome)

    return wav_outcomes

def norm01(data):
    min_value = np.min(data)
    max_value = np.max(data)

    difference = max_value - min_value

    data -= min_value

    if difference > 0:
        data = data / difference

    return data


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    From https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    :param data:
    :param lowcut:
    :param highcut:
    :param fs:
    :param order:
    :return:
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def plot_wavelet_heatmap(threshold,
                         num_levels_contourf, log_y, display_EEG,
                         sampling_rate, wp,
                         wav_outcome, max_v_colorbar,
                         param, file_name,
                         dpi=400, ax_to_use=None,
                         plot_freq_band_detection=True,
                         first_x_tick_label=0,
                         levels=None, red_v_bars=None, save_formats="pdf"):
    """

    :param threshold:
    :param num_levels_contourf:
    :param log_y:
    :param display_EEG:
    :param sampling_rate:
    :param plot_freq_band_detection:
    :param max_v_colorbar : to set the max value from the color bar
    :param length_max_eeg_times:
    :param levels : levels for the colorbar
    :param ax_to_use: if not None, used to plot the spectrogram
    :param red_v_bars: if not None, draw vertical Bar at the x coordinates given as a list of integers
    :return: levels used for contourf if levels is not determined through the parameters
    """
    # used to be plot_wavelet_heatmap_for_a_game in data_analysis

    dbconverted = wav_outcome.dbconverted
    EEGtimes = wav_outcome.eeg_times
    freq_bands = wav_outcome.all_freq_bands
    frex = wav_outcome.frex
    data_bin_for_fft = wav_outcome.data_bin_for_fft
    background_color = "black"
    labels_color = "white"

    length_max_eeg_times = np.max(wav_outcome.eeg_times) - np.min(wav_outcome.eeg_times)
    # print(f"length_max_eeg_times {length_max_eeg_times}, sampling_rate {sampling_rate}")
    # use to set to limit of the colorbar for the heatmap
    max_color_heatmap = np.max(wav_outcome.dbconverted)

    if ax_to_use is None:
        fig = plt.figure(figsize=[16, 4], tight_layout=True, dpi=dpi)
        fig.patch.set_facecolor(background_color)
        height_ratios = [1]
        gs = gridspec.GridSpec(1, 2,
                               width_ratios=[5, 1],
                               height_ratios=height_ratios)
        gs_first_index = 0


        ax1 = plt.subplot(gs[gs_first_index])
    else:
        ax1 = ax_to_use

    ax1.set_facecolor(background_color)

    if log_y:
        ax1.set_yscale("log")

    min_freq = int(np.min(frex))

    max_freq = np.max(frex)
    if (float(max_freq) % 1) >= 0.5:
        max_freq = int(math.ceil(max_freq))
    else:
        max_freq = int(math.floor(max_freq))

    # min_m = np.min(dbconverted)
    # # puting all values positives
    # if min_m < 0:
    #     print('negative values dbconverted')
    #     dbconverted = dbconverted + abs(min_m)
    # # range from 0 to 10
    # dbconverted = dbconverted-np.min(dbconverted)

    # dbconverted = (dbconverted/np.max(dbconverted))*10

    # print(f'min: {np.min(dbconverted)}, max: {np.max(dbconverted)}')

    # threshold according to the one used by Kahana et al. 1999 Nature
    # TODO : Should be the mean + std from all trials powers
    # threshold = np.mean(dbconverted) + np.std(dbconverted)
    # doing that allow the 1 on the scale to be the threshold

    # limit_color_before = np.max(dbconverted) / threshold
    # print(f'threshold {threshold}, np.shape(dbconverted) {np.shape(dbconverted)}')

    dbconverted = dbconverted / threshold

    # num_levels_contourf : represent the automatically-chosen levels that will be contour up to
    limit_color = np.max(dbconverted)  # - 1

    # if baseline_mode:
    #     min_color = limit_color * -1
    # else:
    #     # min_color = ((np.max(dbconverted) - np.min(dbconverted)) / 2) + \
    #     #             ((np.max(dbconverted) - np.min(dbconverted)) / 4)
    #     # if dbconverted if not divided by threshold, then threshold should be use as min_color
    #     min_color = 1  # if dbconverted = dbconverted/threshold, then z = 1
    min_color = 1  # if dbconverted = dbconverted/threshold, then z = 1
    # not used anymore
    max_color = limit_color

    # max_v_colorbar represents the max value from all powers during the different sessions
    if (max_v_colorbar / threshold) > min_color:
        max_v_colorbar = max_v_colorbar / threshold
    try:
        cp = ax1.contourf(EEGtimes, frex, dbconverted, num_levels_contourf,
                          # alpha=0.5,
                          # fix the limits between which the color scale will be done
                          vmin=min_color, vmax=max_v_colorbar, levels=levels,
                          # antialiased=True,
                          # norm=plt_colors.LogNorm(1, np.max(dbconverted)),
                          cmap=plt.cm.jet,  # gnuplot,  #YlOrRd, jet,
                          origin="lower")
    except TypeError:
        print(f"TypeError")
        return
    # plt.contour(EEGtimes, frex, dbconverted, num_levels_contourf,
    #                       alpha=0.8,
    #                       # fix the limits between which the color scale will be done
    #                       vmin=limit_color, vmax=limit_color,
    #                       # antialiased=True,
    #                       linestyles='solid',
    #                       linewidths=0.3,
    #                       # norm=plt_colors.LogNorm(1, np.max(dbconverted)),
    #                       cmap=plt.cm.gist_gray, #binary,
    #                       origin="lower")

    # plot EEG signal
    # scaling for it to fit on a height of 2
    # normalization
    if display_EEG:
        data_bin_plot_eeg = data_bin_for_fft
        band_pass_eeg = False
        if band_pass_eeg:
            lowcut = min_freq
            highcut = max(100, max_freq)
            data_bin_plot_eeg = butter_bandpass_filter(data_bin_plot_eeg, lowcut, highcut, sampling_rate, order=3)
        # print(f"data_bin_plot_eeg {data_bin_plot_eeg}")
        zoom_eeg_factor = 5
        data_bin_plot_eeg = norm01(data_bin_plot_eeg) * zoom_eeg_factor
        # data_bin_plot_eeg = data_bin_plot_eeg + abs(np.min(data_bin_plot_eeg))
        height_from_top = zoom_eeg_factor + 1
        if wp.log_y:
            height_from_top = (max_freq / 10) * 3
        data_bin_plot_eeg = data_bin_plot_eeg + (max_freq - height_from_top)
        # data_bin_plot_eeg = ((data_bin_plot_eeg / np.max(data_bin_plot_eeg)) * 2) + (max_freq - height_from_top)
        ax1.plot(EEGtimes, data_bin_plot_eeg, color="k", lw=0.5)
    # cp = plt.pcolormesh(EEGtimes, frex, dbconverted)

    if log_y:
        ax1.set_yticks(np.logspace(log10(min_freq), log10(max_freq), 6))
        yticks_labels_tmp = np.logspace(log10(min_freq), log10(max_freq), 6)
        yticks_labels = []
        for ytick_label in yticks_labels_tmp:
            ytick_label = np.round(ytick_label, 1)
            if (ytick_label % 1) >= 0.85:
                ytick_label = int(math.ceil(ytick_label))
            elif (ytick_label % 1) <= 0.15:
                ytick_label = int(math.floor(ytick_label))
            yticks_labels.append(ytick_label)
        ax1.set_yticklabels(yticks_labels)


    data_first_index_range = np.min(EEGtimes)
    data_last_index_range = np.max(EEGtimes)

    # tick every 5 seconds
    if (length_max_eeg_times / sampling_rate) > 50:
        step_ticks = 10
    else:
        step_ticks = 5

    ax1.set_xticks(np.arange(data_first_index_range, data_first_index_range + length_max_eeg_times +
                             (step_ticks * sampling_rate),
                             step_ticks * sampling_rate))
    xticklabels = np.arange(first_x_tick_label, first_x_tick_label + (data_last_index_range // sampling_rate) - (data_first_index_range // sampling_rate) +
                            step_ticks,
                            step_ticks).astype("int8")
    # print(f'max_all_eeg_times {max_all_eeg_times}, data_last_index_range {data_last_index_range}')
    # from list of in to list of str
    xticklabels = [str(x) for x in xticklabels]

    ax1.set_xticklabels(
        xticklabels,
        fontsize=10, rotation='vertical')
    xticks = ax1.xaxis.get_major_ticks()
    # remove the ticks that are not used
    if (data_last_index_range - data_first_index_range) < length_max_eeg_times:
        for xtick in xticks[len(xticklabels):]:
            xtick.set_visible(False)
    # ax1.tick_params(axis='x', which='both', length=0)
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_xlabel('Time (sec)')
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    ax1.tick_params(top=False, bottom=True, left='on', right=False, labelleft=True, labelbottom=True)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)


    # # 4, 8 & 13 Hz line
    # for y in [4, 8, 13]:
    #     ax1.plot((data_first_index_range, data_last_index_range), (y, y), linestyle='--', linewidth=1,
    #              color='k')

    # print(f'data_first_index_range {data_first_index_range}, data_last_index_range {data_last_index_range}')
    # allow to alternate color between each freq band
    if plot_freq_band_detection:
        colors_band = ['k', 'w']
        for i_f, f in enumerate(freq_bands):
            # xmin and xmax are comprised between 0 and 1, represent the start from the lines
            # each line represent an episode of a specific frequency_band
            # xmin = (f.starting_point-data_first_index_range) / (data_last_index_range-data_first_index_range)
            # xmax = ((f.starting_point+f.duration)-data_first_index_range) / (data_last_index_range-data_first_index_range)
            # print(f'f.starting_point : {f.starting_point}, f.duration : {f.duration}, ')
            # # print(f'xmin : {xmin}, xmax : {xmax}')
            # # print(f'f.mean_freq() {f.mean_freq()}')
            # plt.axhline(y=f.mean_freq(), linestyle='-', linewidth=1, color=colors_band[i_f % 2], xmin=xmin, xmax=xmax)
            y = f.mean_freq()
            ax1.plot((f.starting_point, (f.starting_point + f.duration)), (y, y), linestyle='-', linewidth=1,
                     color=colors_band[i_f % 2])
            # print(f'len(f.coords) : {len(f.coords)}')

            plot_red_lines = False
            if plot_red_lines:
                # plot for a freq_band, red line for each frequency within from the min to max time for that given freq
                freq_range_index = np.arange(f.min_freq_index, f.max_freq_index, 1)
                for freq_index in freq_range_index:
                    if freq_index in f.min_time_index_for_freq_index:
                        min_time_index = f.min_time_index_for_freq_index[freq_index]
                        max_time_index = f.max_time_index_for_freq_index[freq_index]
                        # print(f'min_time_index: {min_time_index}, max_time_index {max_time_index},
                        # freq_index {freq_index}')
                        min_time = EEGtimes[min_time_index]
                        max_time = EEGtimes[max_time_index]
                        y = frex[freq_index]
                        ax1.plot((min_time, max_time), (y, y), linestyle='-', linewidth=1, color='red')

        # ticks, labels and axis :

    if red_v_bars is not None:
        for t in red_v_bars:
            ax1.axvline(x=t,
                        color='r', linestyle='--', linewidth=2, label='start')

    if wp.detect_spikes:
        for spike in wav_outcome.spikes:
            if (spike > EEGtimes[0]) and (spike < EEGtimes[-1]):
                y_from_bottom = 3
                if wp.log_y:
                    y_from_bottom = 1
                ax1.plot((spike, spike), (1, y_from_bottom), linestyle='-', linewidth=2,
                         color='black')
                # ax1.plot((spike, spike + (wp.spike_removal_time_after * sampling_rate)),
                #          (np.mean([1, y_from_bottom]), np.mean([1, y_from_bottom])),
                #          linestyle='-', linewidth=1,
                #          color='r')
                # ax1.plot((spike - (wp.spike_removal_time_before * sampling_rate), spike),
                #          (np.mean([1, y_from_bottom]), np.mean([1, y_from_bottom])),
                #          linestyle='-', linewidth=1,
                #          color='r')
                height_from_top = 1
                if wp.log_y:
                    height_from_top = (max_freq / 10) * 2
                ax1.plot((spike, spike), (max_freq, max_freq - height_from_top), linestyle='-', linewidth=2,
                         color='black')
                # ax1.plot((spike, spike + (wp.spike_removal_time_after * sampling_rate)),
                #          (max_freq - (height_from_top / 2), max_freq - (height_from_top / 2)),
                #          linestyle='-', linewidth=1,
                #          color='r')
                # ax1.plot((spike - (wp.spike_removal_time_before * sampling_rate), spike),
                #          (max_freq - (height_from_top / 2), max_freq - (height_from_top / 2)),
                #          linestyle='-', linewidth=1,
                #          color='r')

    if ax_to_use is not None:
        return

    cb = plt.colorbar(cp, shrink=1)  # , orientation='horizontal')  # , drawedges=True)
    cb.set_label("(a.u.)")
    cb.ax.tick_params(axis='y', colors="white")

    # power spectrum
    ax3 = plt.subplot(gs[gs_first_index + 1])
    ax3.set_facecolor(background_color)
    index_min = int(np.nonzero(wav_outcome.power_spect_freq == min_freq)[0])
    if len(np.nonzero(wav_outcome.power_spect_freq == max_freq)[0]) == 0:
        index_max = len(wav_outcome.power_spect_freq) - 1
    else:
        index_max = int(np.nonzero(wav_outcome.power_spect_freq == max_freq)[0])
    # if True, use log scale
    log_scale = False
    if log_scale:
        pow_spectrum_to_display = wav_outcome.power_spectrum
    else:
        pow_spectrum_to_display = wav_outcome.power_spectrum
        # pow_spectrum_to_display = 10 * log10(wav_outcome.power_spectrum)
        # wav_outcome.power_spectrum[index_min:index_max] / \
        #                       (np.mean(wav_outcome.power_spectrum[index_min:index_max]) +
        #                        np.std(wav_outcome.power_spectrum[index_min:index_max]))
    # display the freq in y, and pow spectrum in x (

    new_pow_spect = pow_spectrum_to_display[index_min:index_max]
    min_v = new_pow_spect.min()
    if min_v < 0:
        new_pow_spect = np.array([p + abs(min_v) for p in new_pow_spect])
    remove_min = True
    if remove_min:
        min_v = new_pow_spect.min()
        new_pow_spect = np.array([p - min_v for p in new_pow_spect])
    square_it = True
    if square_it:
        new_pow_spect = np.array([p * p for p in new_pow_spect])
    normalize_it = True
    if normalize_it:
        # mean_new_pow_spect = np.mean(new_pow_spect)
        # new_pow_spect = new_pow_spect - mean_new_pow_spect
        std_new_pow_spect = np.std(new_pow_spect)
        new_pow_spect = new_pow_spect / std_new_pow_spect
    ax3.barh(wav_outcome.power_spect_freq[index_min:index_max], new_pow_spect,
             height=1.0, color="blue", log=log_scale)
    # ax3.set_xlim([np.min(pow_spectrum_to_display[index_min:index_max]),
    #               np.max(pow_spectrum_to_display[index_min:index_max])])
    # fixing the label so it looks like it start in negative
    if min_v < 0:
        new = [int(l.get_text()) + min_v if l.get_text().isdigit() else
               l.get_text() for l in ax3.get_yticklabels()]
        ax3.set_xticklabels(new)

    if log_scale:
        ax3.set_xscale('log')
    # ax3.set_yticks(np.arange(0, len(wav_outcome.power_spect_freq), 1))
    # ax3.set_yticklabels(wav_outcome.power_spect_freq)
    # ax3.ylabel('Frequency (Hz)')
    # ax3.xlabel('Time (sec)')
    ax3.tick_params(axis='y', colors=labels_color)
    ax3.tick_params(axis='x', colors=labels_color)
    if log_y:
        ax3.set_yscale("log")
        ax3.set_yticks(np.logspace(log10(min_freq), log10(max_freq), 6))
        ax3.set_yticklabels(yticks_labels)
    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/'
                    f'{file_name}.{save_format}',
                    facecolor=fig.get_facecolor())
    plt.close()
    return cp.levels


# define a freq band signature with name, low and high boundaries of the freq band
class FreqBandId:
    def __init__(self, var_name, name, low_freq, high_freq):
        self.var_name = var_name
        self.name = name
        self.low_freq = low_freq
        self.high_freq = high_freq
        # width of the freq band used to detect, in sec
        # for theta, it would 0.5 Hz as to cover 3 theta wave during that period (using 6Hz)
        self.width_detection = (1 / np.mean([low_freq, high_freq])) * 3


TYPE_OF_FREQ_BAND = dict()
TYPE_OF_FREQ_BAND['delta'] = FreqBandId(var_name='delta', name='delta', low_freq=1, high_freq=5)
# TYPE_OF_FREQ_BAND['theta'] = FreqBandId(var_name='theta', name='theta', low_freq=4, high_freq=8)
# TYPE_OF_FREQ_BAND['delta_theta'] = FreqBandId(var_name='delta_theta', name='delta-theta', low_freq=1, high_freq=8)
# # low and high theta as the study from Bush et al. 2017
# TYPE_OF_FREQ_BAND['low_theta'] = FreqBandId(var_name='low_theta', name='low-theta', low_freq=2, high_freq=5)
# TYPE_OF_FREQ_BAND['high_theta'] = FreqBandId(var_name='high_theta', name='high-theta', low_freq=6, high_freq=9)
# TYPE_OF_FREQ_BAND['alpha'] = FreqBandId(var_name='alpha', name='alpha', low_freq=8, high_freq=13)
# # alpha 1 et 2 as the study of Cragg et al. 2011 that looks at EEG maturation
# TYPE_OF_FREQ_BAND['alpha1'] = FreqBandId(var_name='alpha1', name='alpha1', low_freq=7.5, high_freq=9)
# TYPE_OF_FREQ_BAND['alpha2'] = FreqBandId(var_name='alpha2', name='alpha2', low_freq=9.5, high_freq=12)
# TYPE_OF_FREQ_BAND['theta_alpha'] = FreqBandId(var_name='theta_alpha', name='theta-alpha', low_freq=4, high_freq=13)
TYPE_OF_FREQ_BAND['rodent_theta'] = FreqBandId(var_name='rodent_theta', name='rodent theta', low_freq=5,
                                               high_freq=12)
# TYPE_OF_FREQ_BAND['beta'] = FreqBandId(var_name='beta', name='beta', low_freq=13, high_freq=30)
# TYPE_OF_FREQ_BAND['gamma'] = FreqBandId(var_name='gamma', name='gamma', low_freq=30, high_freq=150)
# TYPE_OF_FREQ_BAND['slow_gamma'] = FreqBandId(var_name='slow_gamma', name='slow-gamma', low_freq=30, high_freq=49)
# TYPE_OF_FREQ_BAND['mid_gamma'] = FreqBandId(var_name='mid_gamma', name='mid-gamma', low_freq=51, high_freq=90)
# TYPE_OF_FREQ_BAND['fast_gamma'] = FreqBandId(var_name='fast_gamma', name='fast-gamma', low_freq=90, high_freq=150)


class GeneralParamaters:
    def __init__(self, path_data, path_results, time_str):
        self.path_data = path_data
        self.path_results = path_results
        self.time_str = time_str


class WaveletParameters:
    def __init__(self,
                 min_cycle, max_cycle, wav_cycles, freq_steps,
                 min_freq, max_freq, num_frex,
                 hz_gap_fb_ep,
                 step_fb_ep,
                 freq_band_to_explore,
                 show_freq_bands,
                 detect_spikes=False,
                 spike_removal_time_after=1,
                 spike_removal_time_before=0,
                 low_freq_spike_detector=3,
                 high_freq_spike_detector=15,
                 compute_freq_bands=False,
                 baseline_duration=120,
                 max_baseline_duration=False,
                 use_derivative=False,
                 increase_gaussian_cycles=False,
                 baseline_mode=None, log_y=False,
                 using_median_for_threshold=False,
                 spike_percentage=0.7,
                 spike_threshold_ratio=1.2,
                 bosc_method=False,
                 welch_method=False,
                 only_compute_baseline_stats=False):
        """

        :param min_cycle:
        :param max_cycle:
        :param wav_cycles:
        :param min_freq:
        :param max_freq:
        :param num_frex:
        :param increase_gaussian_cycles: boolean : define if the number of cycles or the wavelet
        should increase as the frequency do if True, increase from min_cycle to max_cycle a larger number of
        cycles gives you better frequency precision at the cost of worse temporal precision, and a smaller number of
        cycles gives you better temporal precision at the cost of worse frequency precision
        :param log_y: boolean to decide if y-axis and frequency steps are in log scale
        :param use_derivative: if True, will apply numpy.diff to the binary data before applying the wavelet
        :param baseline_duration: in seconds (starting from the beginning of the set
        :param max_baseline_duration: if True, baseline will be as long as data for baseline is available
        :param freq_steps: freq steps between freq to apply by the wavelet, in Hz
        :param baseline_mode: 3 baseline_mode choice : "decibel_conversion", "percentage_change", "z_score_normalization"
        """
        self.baseline_mode = baseline_mode
        # 2 next variables only used if no specific value has been specified for a patient
        self.baseline_duration = baseline_duration
        self.max_baseline_duration = max_baseline_duration
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.num_frex = num_frex
        self.freq_steps = freq_steps
        self.use_derivative = use_derivative
        self.increase_gaussian_cycles = increase_gaussian_cycles
        self.min_cycle = min_cycle
        self.max_cycle = max_cycle
        self.wav_cycles = wav_cycles
        self.log_y = log_y
        self.hz_gap_fb_ep = hz_gap_fb_ep
        self.step_fb_ep = step_fb_ep
        self.using_median_for_threshold = using_median_for_threshold
        self.time_str = datetime.now().strftime("%Y_%m_%d.%H-%M-%S")
        self.compute_freq_bands = compute_freq_bands
        self.spike_percentage = spike_percentage
        self.spike_threshold_ratio = spike_threshold_ratio
        self.low_freq_spike_detector = low_freq_spike_detector
        self.high_freq_spike_detector = high_freq_spike_detector
        self.spike_removal_time_after = spike_removal_time_after
        self.spike_removal_time_before = spike_removal_time_before
        self.detect_spikes = detect_spikes
        self.bosc_method = bosc_method
        self.welch_method = welch_method
        # if True, baseline will be computed using 1/f
        self.only_compute_baseline_stats = only_compute_baseline_stats
        self.show_freq_bands = show_freq_bands

        # dict that will contain the string representation for a variable as key, and as value the value of the variable
        self.var_num_dict = dict()
        self.var_num_dict["Minimum freq"] = self.min_freq
        self.var_num_dict["Maximum freq"] = self.max_freq
        self.var_num_dict["Frequency steps"] = self.freq_steps
        self.var_num_dict["Number of frequencies"] = self.num_frex
        self.var_num_dict["Minimum number of cycles"] = self.min_cycle
        self.var_num_dict["Maximum number of cycles"] = self.max_cycle
        self.var_num_dict["Number of wavelet cycles"] = self.wav_cycles
        self.var_num_dict["Hz gap between overlap episode band"] = self.hz_gap_fb_ep
        self.var_num_dict["Steps to detect episode band"] = (self.step_fb_ep * self.freq_steps)
        self.var_num_dict["Spike threshold ratio"] = self.spike_threshold_ratio
        # for next update
        # self.var_num_dict["Spike vertical percentage"] = self.spike_percentage
        self.var_num_dict["Lowest freq use for spike detector"] = self.low_freq_spike_detector
        self.var_num_dict["Highest freq use for spike detector"] = self.high_freq_spike_detector
        self.var_num_dict["Spike removal time after"] = self.spike_removal_time_after
        self.var_num_dict["Spike removal time before"] = self.spike_removal_time_before

        # Remove freq band that are out of the frequency range
        filtered_freq_band_to_explore = []
        for fbi_str in freq_band_to_explore:
            if fbi_str in TYPE_OF_FREQ_BAND:
                fbi = TYPE_OF_FREQ_BAND[fbi_str]
            else:
                continue
            if fbi.high_freq <= self.min_freq:
                continue
            if fbi.low_freq >= self.max_freq:
                continue
            filtered_freq_band_to_explore.append(fbi_str)

        self.freq_band_to_explore = filtered_freq_band_to_explore

    def __str__(self):
        new_line = "\n"
        result = ""
        if self.baseline_mode:
            if self.max_baseline_duration:
                result += "Maximal baseline duration"
            else:
                result += f"Default baseline duration: {self.baseline_duration}"
            result += new_line
        if self.use_derivative:
            result += "Using 1/f"
            result += new_line
        if self.increase_gaussian_cycles:
            result += "Increasing gaussian cycles"
            result += new_line
        if self.using_median_for_threshold:
            result += "Threshold using median value"
        else:
            result += "Threshold using mean value + std"
        result += new_line
        if self.log_y:
            result += "Log Y values"
            result += new_line
        if self.remove_spikes:
            result += "Spikes removed"
            result += new_line
        if self.bosc_method:
            result += "BOSC method applied"
            result += new_line
        result += f"Freq bands explored: {self.freq_band_to_explore}"
        result += new_line

        for key, value in self.var_num_dict.items():
            result += f"{key}: {value}"
            result += new_line

        return result

    def same_wp_from_file(self, file_name, end_tag):
        """
        Will read the file filename till the end_tag, and will load the information corresponding to str(wp),
        and check if the configuraiton is the same as self
        :param file_name:
        :param end_tag: a string, like "###' to know when to stop
        :return:
        """
        log_y = False
        with open(file_name, "r", encoding='UTF-8') as file:
            # first lines of the file present the configuration
            for line_index, line in enumerate(file):
                # first line if the date of the day the file was registered
                if line_index == 0:
                    continue
                if end_tag in line:
                    return True
                if line.endswith("\n"):
                    line = line[:-1]
                line = line.split(":")
                # print(f'line {line}')
                if len(line) == 1:
                    line = line[0]
                    if line == "\n":
                        continue
                    if line == "Using 1/f":
                        if not self.use_derivative:
                            return False
                        continue
                    if line == "Maximal baseline duration":
                        if not self.max_baseline_duration:
                            return False
                        continue
                    if line == "Increasing gaussian cycles":
                        if not self.increase_gaussian_cycles:
                            return False
                        continue
                    if line == "Threshold using median value":
                        if not self.using_median_for_threshold:
                            return False
                        continue
                    if line == "Threshold using mean value + std":
                        if self.using_median_for_threshold:
                            return False
                        continue
                    if line == "Log Y values":
                        log_y = True
                        if not self.log_y:
                            return False
                        continue
                    if line == "Spikes removed":
                        if not self.remove_spikes:
                            return False
                        continue
                    if line == "BOSC method applied":
                        if not self.bosc_method:
                            return False
                        continue
                    continue
                if len(line) > 2:
                    return False
                var_str = line[0]
                if var_str == "Freq bands explored":
                    if line[1][1:] != str(self.freq_band_to_explore):
                        return False

                # then we gather numerical value on the right
                try:
                    num = int(line[1])
                except ValueError:
                    # it means it's not a number
                    continue

                if var_str in self.var_num_dict:
                    if self.var_num_dict[var_str] != num:
                        # print(f'self.var_num_dict[var_str] != num')
                        return False
        if self.log_y and not log_y:
            return False
        return True

    def html_descr(self):
        """
        Return an html description of the paramaters
        :return:
        """
        new_line = "<br>\n"
        html_str = ""
        html_str += "<div style=\"text-align:center;border:1px solid red\"> <b>Wavelet parameters</b></div>"
        html_str += new_line
        if self.baseline_mode:
            if self.max_baseline_duration:
                html_str += "Maximal baseline duration"
            else:
                html_str += f"Default baseline duration: {self.baseline_duration}"
            html_str += new_line
        if self.use_derivative:
            html_str += "Using 1/f"
            html_str += new_line
        if self.only_compute_baseline_stats:
            html_str += "Only compute baseline stats"
            html_str += new_line
        if self.increase_gaussian_cycles:
            html_str += "Increasing gaussian cycles"
            html_str += new_line
        if self.remove_spikes:
            html_str += "Spikes removed"
            html_str += new_line
        if self.using_median_for_threshold:
            html_str += "Threshold using median value"
        else:
            html_str += "Threshold using mean value + std"
        html_str += new_line
        if self.bosc_method:
            html_str += "BOSC method applied"
            html_str += new_line
        html_str += new_line
        if self.log_y:
            html_str += "Log Y values"
            html_str += new_line

        html_str += f"Freq bands explored: {self.freq_band_to_explore}"
        html_str += new_line

        for key, value in self.var_num_dict.items():
            html_str += f"{key}: {value}"
            html_str += new_line

        return html_str


class WaveletOutcome:
    """
    Represent the result of a wavelet on a databin with dataset necessary to analyse it
    """

    def __init__(self, eeg_times, frex, data_bin_for_fft, sampling_rate,
                 power_spect_freq, power_spectrum, max_value_dbconverted, events_time_stamp_hz=None,
                 title="", is_placing=None, session_index=None, dbconverted=None):
        self.dbconverted = dbconverted
        self.eeg_times = eeg_times
        self.frex = frex
        self.data_bin_for_fft = data_bin_for_fft
        self.events_time_stamp_hz = events_time_stamp_hz
        self.title = title
        self.is_placing = is_placing
        self.session_index = session_index
        self.sampling_rate = sampling_rate
        self.threshold = None
        self.power_spect_freq = power_spect_freq
        self.power_spectrum = power_spectrum
        self.max_value_dbconverted = max_value_dbconverted
        self.all_freq_bands = []
        self.spikes = []
        self.original_spikes = []


class FrequencyBandEpisode:
    """
    Represent a frequency band episode
    """

    def __init__(self, freq_limits, duration, starting_point, sampling_rate, **kwargs):
        """

        :param freq_limits: list of 2 integers
        :param duration: time duration
        :param starting_point: time point (from the start of the ampli)
        :param freq_indexes : set of freq_index and time_index
        """
        self.freq_limits = freq_limits
        self.duration = duration  # in Hz
        self.starting_point = starting_point
        self.end_point = starting_point + duration
        self.sampling_rate = sampling_rate

    def __str__(self):
        return f'{self.freq_limits[0]} {self.freq_limits[1]} {self.duration} {self.starting_point} {self.sampling_rate}'


class FrequencyBandEpisodeBosc(FrequencyBandEpisode):

    def __init__(self, freq_limits, duration, starting_point, sampling_rate, **kwargs):
        """

        :param freq_limits: list of 2 integers, freq band in which is contains this episode
        :param duration: time duration
        :param starting_point: time point (from the start of the ampli)
        :param freq_indexes : set of freq_index and time_index
        """
        FrequencyBandEpisode.__init__(self, freq_limits, duration, starting_point, sampling_rate)
        self.freq = None
        if kwargs is not None:
            if "freq" in kwargs:
                self.freq = kwargs["freq"]

    def mean_freq(self):
        return self.freq



