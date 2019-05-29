import numpy as np
from pattern_discovery.tools.signal import gaussblur1D, norm01, gauss_blur
from pattern_discovery.display.raster import plot_with_imshow
from skimage.filters import threshold_otsu
import scipy
import sys
from sklearn.decomposition import PCA
import os


def xcov(a, b, lags):
    """
    Cross-covariance or autocovariance, returned as a vector or matrix.

    If x is an M × N matrix, then xcov(x) returns a (2M – 1) × N2 matrix
    with the autocovariances and cross-covariances of the columns of x. If you specify a maximum lag maxlag, then the output c has size (2 × maxlag – 1) × N2.
    :param a: 1st matrix
    :param b: 2nd matrix
    :param lags:
    :return:
    """

    # the cross-covariance is the cross-correlation function of two sequences with their means removed
    a = a - np.mean(a)
    b = b - np.mean(b)
    # with full mode, test all the lags
    xy_cov = np.correlate(a, b, mode="full")
    center_index = len(xy_cov) // 2
    if lags <= 0:
        return xy_cov[center_index]
    else:
        return xy_cov[center_index-lags:center_index+lags+1]


def covnorm(a, b, lags):
    """
    Compute the normalized cross-covariance
    :param a:
    :param b:
    :param lags:
    :return:
    """
    n = len(a)
    x_cov_result = xcov(a, b, lags)
    # print(f"x_cov_result {x_cov_result}")
    c = x_cov_result / np.std(a) / np.std(b) / n
    return c


def step1_pca(data, n_pc_max):
    """

    :param data: matrix n_cells x n_times -
    :param n_pc_max: scalar - number max of principal components
    :return: data_new: n_cells x n_times centralized movie image used for PCA
    p_cout dict with filed pc_xfilter (n_cells x n_pc) spatial eigenvectors,
    pc_time_course (n_times, n_pc) temporal eigenvectors
    pc_eigval (n_pc) eigenvalues
    """

    # number of cells and times
    n_cells, n_times = data.shape

    # centralize data
    # average value for each cell
    cells_mean = np.mean(data, axis=1)
    cells_mean_2d = np.repeat(cells_mean, n_times, axis=0)
    cells_mean_2d = cells_mean_2d.reshape((n_cells, n_times))
    # centralized (temporal)
    data_centralized = (data - cells_mean_2d)

    # average instensity at each time frame - average over all pixels
    times_mean = np.mean(data, axis=0)
    times_mean_2d = np.repeat(times_mean, n_cells, axis=0)
    times_mean_2d = times_mean_2d.reshape((n_times, n_cells)).transpose()
    data_centralized = (data_centralized - times_mean_2d)

    data = data_centralized

    # Singular Value Decomposition
    # M = UDV' eigenvectors U: spatial filters, V: signal time courses
    # calculate eigenvectors on smaller dimension, then compute the eigenvectors in the complementary space
    # (temporal, spatial)
    # U and V are both orthonormal

    if n_times < n_cells:
        # then we would need to do spatial covariance on temporal matrix
        raise Exception("n_times < n_cells")

    data_mt = (np.dot(data[:, :-1].transpose(), data[:, 1:]) + np.dot(data[:, 1:].transpose(), data[:, :-1]))\
              / n_times / 2
    print(f"data_mt {data_mt.shape}")
    # eigenvectors and eigenvalues
    u_all, eigval_all = scipy.sparse.linalg.eigs(data_mt, n_pc_max)
    # eigval_all = np.real(eigval_all)
    print(f"eigval_all shape {eigval_all.shape}")
    # keep positive non-zero eig val
    pos = np.where(eigval_all > sys.float_info.epsilon)
    eigval = eigval_all[:, np.unique(pos[1])]
    eigval = eigval[np.unique(pos[0]), :]

    print(f"eigval {eigval.shape}")

    # keep corresponding eigenvectors
    print(f"u_all {u_all.shape}")
    u = u_all[np.unique(pos[1])]
    print(f"u {u.shape}")

    # find V
    # print(f"eigval {eigval}")
    mat_svd = np.sqrt(n_times * eigval)
    # print(f"eig_val {eig_val.shape}")
    print(f"mat_svd {mat_svd.shape}")
    # print(f"u_all {u_all.shape}")
    least_squares_solution = np.linalg.lstsq(mat_svd.transpose(), u)[0]
    print(f"least_squares_solution {least_squares_solution.shape}")
    v = (least_squares_solution * data_mt).transpose()

    pc_cout = dict()
    pc_cout["pc_x_filter"] = u
    pc_cout["pc_time_course"] = v
    pc_cout["pc_eig_val"] = np.diag(eigval)

    # data has been normalized
    return data, pc_cout

def find_seq_with_pca(traces, path_results, file_name, speed=None):
    if traces is None:
        print(f"traces should not be None")
        return

    n_cells, n_times = traces.shape
    dt = 200
    np_pc_max = 10
    arnaud_version = True
    if arnaud_version:
        m_4, p_cout = step1_pca(gaussblur1D(traces, n_times/20, 1), np_pc_max)
        pc_time_course = np.real(p_cout["pc_time_course"])  #.transpose()
    # else:
    #     pca = PCA(n_components=np_pc_max)  #
    #     pca_result = pca.fit_transform(gaussblur1D(traces, n_times/20, 1))
    #     # print(f"pca_result.shape {pca_result.shape}")
    #     # print(f"pca_result[0,:] {pca_result[0,:]}")
    #     # print(f"pca_result[1,:] {pca_result[1,:]}")
    #     # print(f"pca.explained_variance_ratio_ {pca.explained_variance_ratio_}")
    #     for component in np.arange(np_pc_max):
    #         # sorted_raster = np.copy(spike_nums_dur)
    #         indices_sorted = np.argsort(pca_result[:, component])

    print(f"pc_time_course {pc_time_course.shape}")
    for pc_number in np.arange(np_pc_max):
        sig_ref = np.diff(pc_time_course[pc_number, :])

        # calculate shifted trace
        cor_1d = np.zeros(n_cells)
        shift_1d = np.zeros(n_cells)
        thresholds = np.zeros((n_cells, n_times))
        # print(f"sig_ref {sig_ref}")
        for i in np.arange(n_cells):
            tmp = covnorm(traces[i, :], np.real(sig_ref), dt)
            # print(f"tmp {tmp}")
            cor_1d[i] = np.max(tmp)
            # print(f"np.where(tmp == cor_1d[i])[0] {np.where(tmp == cor_1d[i])[0]}")
            shift_1d[i] = np.where(tmp == cor_1d[i])[0][0] - dt - 1
            # print(f"shift_1d[i] {shift_1d[i]}")
            thresholds[i] = np.roll(np.transpose(traces[i, :]), - int(shift_1d[i]))

        # dc == distance cell
        dc = np.where(cor_1d > threshold_otsu(cor_1d))[0]
        n_dc = len(dc)
        traces_dc = traces[dc, :]

        for i in np.arange(n_dc):
            traces_dc[i, :] = norm01(gaussblur1D(traces_dc[i, :], n_times/2, 0))
            traces_dc[i, :] = norm01(traces_dc[i, :])
            traces_dc[i, :] = traces_dc[i, :] - np.median(traces_dc[i, :])

        shift_dc = shift_1d[dc]
        # xDel in the code of Arnaud
        sorted_indices = np.argsort(shift_dc)

        if speed is not None:
            speed_blur = gauss_blur(speed, n_times/10)

        t = np.arange(n_times) / 10

        print("find_seq_with_pca done")
        plot_with_imshow(raster=traces_dc[sorted_indices], path_results=path_results,
                         file_name=file_name + f"_pc_{pc_number}", n_subplots=4,
                         values_to_plot=None, cmap="hot", show_fig=True, save_formats="pdf")

        with open(os.path.join(path_results, file_name + f"_pc_{pc_number}.txt"), "w", encoding='UTF-8') as file:
            file.write(f"Sorted cells" + '\n')
            for i in sorted_indices:
                file.write(f"{i}")
                if i < len(sorted_indices) - 1:
                    file.write(" ")
            file.write(f"" + '\n')