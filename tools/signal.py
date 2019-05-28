import numpy as np

def smooth_convolve(x, window_len=11, window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')

    return y


def gaussblur1D(data, dw, dim):
    n_times = data.shape[dim]

    if n_times % 2 == 0:
        kt = np.arange((-n_times / 2) + 0.5, (n_times / 2) - 0.5 + 1)
    else:
        kt = np.arange(-(n_times - 1) / 2, ((n_times - 1) / 2 + 1))

    if dim == 0:
        fil = np.exp(-np.square(kt) / dw ** 2) * np.ones(data.shape[0])
    elif dim == 1:
        fil = np.ones(data.shape[1]) * np.exp(-np.square(kt) / dw ** 2)
    elif dim == 2:
        fil = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        for i in np.arange(n_times):
            fil[:, :, i] = np.ones((data.shape[0], data.shape[1])) * np.exp(-np.square(kt[i]) / dw ** 2)

    tfA = np.fft.fftshift(np.fft.fft(data, axis=dim))
    b = np.real(np.fft.ifft(np.fft.ifftshift(tfA * fil), axis=dim))

    return b


def gauss_blur(data, dw):
    """
    2D gaussian filter in Fourier domainon 2 first dimensions
    :param data:
    :param dw:
    :return:
    """

    n_x, n_y = data.shape

    if n_x % 2 == 0:
        kx = np.arange((-n_x / 2) + 0.5, (n_x / 2) - 0.5 + 1)
    else:
        kx = np.arange(-(n_x - 1) / 2, ((n_x - 1) / 2 + 1))

    if n_y % 2 == 0:
        ky = np.arange((-n_y / 2) + 0.5, (n_y / 2) - 0.5 + 1)
    else:
        ky = np.arange(-(n_y - 1) / 2, ((n_y - 1) / 2 + 1))

    k_x, k_y = np.meshgrid(kx, ky)

    kr_ho = np.sqrt(np.square(k_x) + np.square(k_y))
    fil = np.exp(-np.square(kr_ho) / dw ** 2)
    # b = np.zeros(n_x, n_y)
    # tfa = np.zeros(n_x, n_y)

    tfA = np.fft.fftshift(np.fft.fft2(data))
    b = np.real(np.fft.ifft2(np.fft.ifftshift(tfA * fil)))

    return b

def norm01(data):
    min_value = np.min(data)
    max_value = np.max(data)

    difference = max_value - min_value

    data -= min_value

    if difference > 0:
        data = data / difference

    return data