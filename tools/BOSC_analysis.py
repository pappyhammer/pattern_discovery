import numpy as np
import scipy.stats as scipystat
from numpy.fft import fft, ifft


def nextpow2(i):
    n = 1
    while n < i :
        n *= 2
    return n


def bosc_detect(b, powthresh, durthresh):
    # % detected=BOSC_detect(b,powthresh,durthresh,Fsample)
    # %
    # % This function detects oscillations based on a wavelet power
    # % timecourse, b, a power threshold (powthresh) and duration
    # % threshold (durthresh) returned from BOSC_thresholds.m.
    # %
    # % It now returns the detected vector which is already episode-detected.
    # %
    # % b - the power timecourse (at one frequency of interest)
    # %
    # % durthresh - duration threshold in  required to be deemed oscillatory
    # % powthresh - power threshold
    # %
    # % returns:
    # % detected - a binary numpy array containing the value 1 for times at
    # %            which oscillations (at the frequency of interest) were
    # %            detected and 0 where no oscillations were detected.
    # % H        - a matrix of 2 dimension, H[0,:] representing the index
    #              of the beginning of a freq band in the vector b, while H[0,:] represent the end index
    # %
    # % NOTE: Remember to account for edge effects by including
    # % "shoulder" data and accounting for it afterwards!
    # %
    # % To calculate Pepisode:
    # % Pepisode=length(find(detected))/(length(detected));
    #
    #t=(1:len(b))/Fsample;
    # number of time points
    nT = len(b)-1

    # Step 1: power threshold
    # converting False and True to 0 and 1, to use np.diff
    x = (b > powthresh).astype(int)
    dx = np.diff(x)
    # show the +1 and -1 edges
    pos = np.where(dx == 1)[0] + 1
    neg = np.where(dx == -1)[0] + 1

    # % now do all the special cases to handle the edges
    detected = np.zeros(len(b), dtype=np.int8)
    # print(f'1 nT: {nT}, powthresh: {powthresh}, durthresh: {durthresh}')
    # print(f'1 x[:20]: {x[:20]}, dx[:20]: {dx[:20]}')
    # print(f'1 len(pos): {len(pos)}, len(neg) {len(neg)}')
    if (pos.size == 0) and (neg.size == 0):
        if len(np.nonzero(x)[0]) > 0:
            H = np.matrix(f'0;{nT}') # all episode
        else:
            H = np.matrix('') # or none
    elif pos.size == 0:
        # i.e., starts on an episode, then stops
        H = np.matrix(f'0;{neg[0]}')
    elif neg.size == 0:
        # starts, then ends on an ep.
        H = np.matrix(f'{pos[0]};{nT}')
        # print(f'neg.size == 0, H: {H}')
    else:
        if pos[0] > neg[0]:
            # we start with an episode
            pos = np.insert(pos, 0, 0)
        if neg[-1] < pos[-1]:
            #  we end with an episode
            neg = np.append(neg, nT)
        # NOTE: by this time, length(pos)==length(neg), necessarily
        H = np.matrix([pos, neg])

    # special-casing, making the H double-vector
    # more than one "hole", True if not empty
    if np.any(H):
        # find epochs lasting longer than minNcycles*period, array of position
        goodep = np.flatnonzero(np.array(H[1, :] - H[0, :]).flatten() >= durthresh)
        if goodep.size == 0:
            H = []
        else:
            H = H[:, goodep]
            #  OR this onto the detected vector
            for h in np.arange(0, H[1, :].size):
                detected[H[0, h]:H[1, h]] = 1

    return detected, H


def bosc_tf(data_bin_for_fft, wp, patient=None, sampling_rate=None):
    #% This function computes a continuous wavelet (Morlet) transform on
    # % a segment of EEG signal; this can be used to estimate the
    # % background spectrum (BOSC_bgfit) or to apply the BOSC method to
    # % detect oscillatory episodes in signal of interest (BOSC_detect).
    # %
    # % parameters:
    # % eegsignal - a row vector containing a segment of EEG signal to be
    # %             transformed
    # % F - a set of frequencies to sample (Hz)
    # % Fsample - sampling rate of the time-domain signal (Hz)
    # % wavenumber is the size of the wavelet (typically, width=6)
    # %
    # % returns:
    # % B - time-frequency spectrogram: power as a function of frequency
    # %     (rows) and time (columns)
    # % T - vector of time values (based on sampling rate, Fsample)
    if sampling_rate is None:
        if patient is None:
            raise Exception("patient can not be None if sampling_rate is None")
        sampling_rate = patient.EEG_info.get_sampling()
    # define wavelet parameters
    # time_wav define how long is the wavelet
    # If time (max-min) is too low, then if we have a min_cycle quite high, the wavelet will not taper to zero
    time_wav = np.arange(-2, 2 + 1 / sampling_rate, 1 / sampling_rate)
    if wp.log_y:
        # the peak frequencies of wavelets increase logarithmically
        frex = np.logspace(np.log10(wp.min_freq), np.log10(wp.max_freq), wp.num_frex)
    else:
        # the peak frequencies of wavelets increase linearly
        frex = np.linspace(wp.min_freq, wp.max_freq, num=wp.num_frex)
    # compute gaussian
    # the number of cycles increases from min_cycle to max_cycle in the same number of steps used
    # to increase the frequency of the wavelets from min_freq Hz to max_freq H
    if wp.increase_gaussian_cycles:
        if wp.log_y:
            s = np.logspace(np.log10(wp.min_cycle), np.log10(wp.max_cycle), wp.num_frex) / (2 * np.pi * frex)
        else:
            s = np.linspace(wp.min_cycle, wp.max_cycle, wp.num_frex) / (2 * np.pi * frex)
    else:
        s = wp.wav_cycles / (2 * np.pi * frex)
    # define convolution parameters
    # print(f'len wav {len(time_wav)}')
    n_wavelet = len(time_wav)
    n_data = len(data_bin_for_fft)  # data_last_index_range - data_first_index_range
    n_convolution = n_wavelet + n_data - 1
    n_conv_pow2 = nextpow2(n_convolution)
    half_of_wavelet_size = (n_wavelet - 1) // 2

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

        # average power over trials
        # this performs baseline transform, which is covered in more depth in ch18
        # decibel- normalized power
        # has been modified comparing to the book
        temppower[fi, :] = np.absolute(eegconv) ** 2

    return temppower, frex, n_convolution


def bosc_thresholds(Fsample, percentilethresh, numcyclesthresh, F, meanpower):
    # %
    # % This function calculates all the power thresholds and duration
    # % thresholds for use with BOSC_detect.m to detect oscillatory episodes
    # % Fsample - sampling rate (Hz)
    # % percentilethresh - power threshold expressed as a percentile/100
    # %                    (i.e., from 0-1) of the estimated
    # %                    chi-square(2) probability distribution of
    # %                    power values. A typical value is 0.95
    # % numcyclesthresh - duration threshold. A typical value is 3 cycles.
    # %
    # % returns:
    # % power thresholds and duration thresholds
    # %
    # % Fsample = is the sampling rate
    # %
    # % F - frequencies sampled in the power spectrum
    # %
    # % meanpower - power spectrum (mean power at each frequency)

    # % power threshold is based on a chi-square distribution with df=2
    # % and mean as estimated previously (BOSC_bgfit.m)
    powthresh = scipystat.chi2.ppf(percentilethresh, 2) * (meanpower / 2)
    # powthresh=chi2inv(percentilethresh,2)*meanpower/2
    # % chi2inv.m is part of the statistics toolbox of Matlab and Octave

    # % duration threshold is simply a certain number of cycles, so it
    # % scales with frequency
    durthresh = np.transpose((numcyclesthresh * Fsample) / F)
    return powthresh, durthresh


def bosc_bgfit(F,B):
    # % [pv,meanpower]=BOSC_bgfit(F,B)
    # %
    # % This function estimates the background power spectrum via a
    # % linear regression fit to the power spectrum in log-log coordinates
    # %
    # % parameters:
    # % F - vector containing frequencies sampled
    # % B - matrix containing power as a function of frequency (rows) and
    # % time). This is the time-frequency data.
    # %
    # % returns:
    # % pv = contains the slope and y-intercept of regression line
    # % meanpower = mean power values at each frequency
    # %
    # linear regression
    # using np.transpose only if no complex data, otherwise if complex data necessity to use conj().transpose()
    # to get the complex conjugate transpose that also negates the sign of the imaginary part of the complex elements
    # in V.
    # pv = np.polyfit(np.log10(F), np.transpose(np.mean(np.log10(B), 2)), 1)

    pv = np.polyfit(np.log10(F), np.mean(np.log10(B), 1).conj().transpose(), 1)

    # transform back to natural units (power; usually uV^2/Hz)
    meanpower = np.power(10, (np.polyval(pv, np.log10(F))))
    return pv, meanpower


def complex_morlet_wavelet(sampling_rate, frequency, s_wav=None, time_wav=None):
    # create wavelet

    # define how long is the wavelet
    if time_wav is None:
        time = np.arange(-1, 1 + 1 / sampling_rate, 1 / sampling_rate)
    else:
        time = time_wav
    # frequency in Hz
    f = frequency
    sine_wave = np.exp(2 * 1j * np.pi * f * time)

    # compute gaussian

    # n : The number of cycles of the Gaussian taper defines its width, which in turn defines the width of the
    # wavelet
    # a larger number of cycles gives you better frequency precision at the cost of worse temporal precision,
    # and a smaller number of cycles gives you better temporal precision at the cost of worse frequency precision
    if s_wav is None:
        n = 4
        s = n / (2 * np.pi * f)
    else:
        s = s_wav
    gaussian_win = np.exp(-time ** 2 / (2 * s ** 2))
    A = np.sqrt(1 / (s * np.sqrt(np.pi)))
    # window the sinewave by a gaussian to create complex morlet wavelet
    wavelet = A * sine_wave * gaussian_win

    # to check if the wavelet taper to zero
    # plt.plot(time, np.real(wavelet))
    # plt.show()
    return wavelet

# def bosc_analysis_on_data_bin(patient, data_bin_for_fft, wp, frex_to_analyse):
#     temppower, frex, n_convolution = bosc.bosc_tf(patient=patient, data_bin_for_fft=data_bin_for_fft, wp=wp)
#     pv, meanpower = bosc.bosc_bgfit(frex, temppower)
#     powthresh, durthresh = bosc.bosc_thresholds(sampling_rate, 0.95, 3, frex, meanpower)
#     for freq in frex_to_analyse:
#         index_freq = np.searchsorted(frex, freq, side="left")
#         detected, h = bosc.bosc_detect(temppower[index_freq, :], powthresh[index_freq], durthresh[index_freq])
#         p_episode = len(np.flatnonzero(detected)) / len(detected)
