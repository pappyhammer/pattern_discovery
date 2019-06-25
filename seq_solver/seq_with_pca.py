import numpy as np
from pattern_discovery.tools.signal import gaussblur1D, norm01, gauss_blur
from pattern_discovery.display.raster import plot_spikes_raster, plot_with_imshow
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
    times_mean = np.mean(data_centralized, axis=0)
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

    data_mt = (np.dot(data[:, :-1], data[:, 1:].transpose()) + np.dot(data[:, 1:], data[:, :-1].transpose()))\
              / n_times / 2
    print(f"data_mt {data_mt.shape}")
    # eigenvectors and eigenvalues
    # eigval_all, eigvectors_all = scipy.sparse.linalg.eigs(data_mt, n_pc_max)
    pca = PCA(n_components=n_pc_max)  #
    pca_result = pca.fit_transform(data_mt)
    eigvectors_all = pca.components_.transpose()
    eigval_all = pca.explained_variance_
    # create a diagonal matrix from eigen_values
    eigval_all = np.diag(eigval_all)
    # eigval_all = np.real(eigval_all)
    print(f"eigval_all shape {eigval_all.shape}")
    # keep positive non-zero eig val
    # pos = np.where(eigval_all > sys.float_info.epsilon)
    # # eigval = eigval_all[:, np.unique(pos[1])]
    # # eigval = eigval[np.unique(pos[0]), :]
    # print(f"pos {' '.join(map(str, pos))}")
    # eigval = eigval_all[pos[0], pos[1]]
    eigval = eigval_all

    print(f"eigval {eigval.shape}")

    # keep corresponding eigenvectors
    print(f"eigvectors_all {eigvectors_all.shape}")
    # eigvectors = eigvectors_all[:, pos[1]]
    eigvectors = eigvectors_all
    print(f"eigvectors {eigvectors.shape}")

    # find V
    # print(f"eigval {eigval}")
    mat_svd = np.sqrt(n_times * eigval)
    # print(f"eig_val {eig_val.shape}")
    print(f"mat_svd {mat_svd.shape}")
    # print(f"u_all {u_all.shape}")
    least_squares_solution = np.linalg.lstsq(mat_svd, eigvectors.transpose())[0]
    print(f"least_squares_solution {least_squares_solution.shape}")
    pc_time_course = np.dot(least_squares_solution, data_mt)

    pc_cout = dict()
    pc_cout["pc_x_filter"] = eigvectors
    pc_cout["pc_time_course"] = pc_time_course
    pc_cout["pc_eig_val"] = np.diag(eigval)

    # data has been normalized
    return data, pc_cout

def find_seq_with_pca(ms, traces, path_results, file_name, speed=None):
    if traces is None:
        print(f"traces should not be None")
        return

    n_cells, n_times = traces.shape

    if ms.pca_seq_cells_order is not None:
        pc3 = "151 51 293 46 67 90 93 60 165 231 283 108 132 117 140 58 155 186 29 10 254 100 45 298 217 86 61 163 68 125 52 53 32 89 109 213 180 122 27 147 207 3 126 302 202 178 92 99 297 118 247 130 31 72 104 103 189 269 203 101 185 78 168 195 184 226 21 303 15 129 239 66 121 187 81 115 20 172 138 216 237 182 28 223 149 280 229 246 120 176 83 292 198 128 173 188 263 54 296 112 13 6 16 0 281 222 136 26 190 44 245 224 102 17 288 193 48 23 256 143 158 157 43 36 167 74 299 209 22 191 85 250 261 42 82 276 201 5 127 95 242 274 275 148 152 39 14 7 139 40 71 64 249 18 135 253 206 289 208 272 277 19 69 290 65 70 279 76 11 199 80 8 257 241 260 1 2 4 144 75 59 232 194 84 234 240 49 170 62 98 153 175 181 134 215 244 160 131 197 300 219 270 278 97 221 225 220 212 156 255 164 236 106 169 57 179 271 304248 9 287 285 88 114 30 259 110 91 192 171 211 228 258 55 105 301 204 124 24 63 142 262 196 218 238 96 282 235 251 77 137 227 56 145 200 119 111 230 162 154 79 273 205 12 34 133 214 161 252 47 243 286 38 266 166 37 94 146 294 267 50 174 210 268 233 116 183 264 291 159 123 25 107 150 141 113 265 87 177 41 35 73 33 284 295"
        pc3 = [int(x) for x in pc3.split()]
        print(f"len(pc3) {len(pc3)}")
        print(f"find_seq_with_pca for {ms.description} using matlab results")
        print(f"pca_seq_cells_order: {len(ms.pca_seq_cells_order)}")
        print(f"len diff: {len(np.setdiff1d(ms.pca_seq_cells_order, pc3))}")
        traces_dc = traces[ms.pca_seq_cells_order, :]

        for i in np.arange(len(traces_dc)):
            traces_dc[i, :] = norm01(gaussblur1D(traces_dc[i, :], n_times / 2, 0))
            traces_dc[i, :] = norm01(traces_dc[i, :])
            traces_dc[i, :] = traces_dc[i, :] - np.median(traces_dc[i, :])
        plot_with_imshow(raster=traces_dc, path_results=path_results,
                         y_ticks_labels_size=0.1,
                         x_ticks_labels_size=5,
                         y_ticks_labels=ms.pca_seq_cells_order,
                         file_name=file_name + f"_matlab_version",
                         n_subplots=4,
                         without_ticks=False,
                         vmin=0, vmax=0.5, hide_x_labels=False,
                         values_to_plot=None, cmap="hot", show_fig=False,
                         save_formats="pdf")
        return

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
                         y_ticks_labels_size=0.1,
                         y_ticks_labels=sorted_indices,
                         without_ticks=False,
                         x_ticks_labels_size=5,
                         vmin=0, vmax=0.5, hide_x_labels=False,
                         file_name=file_name + f"_pc_{pc_number}",
                         n_subplots=4,
                         values_to_plot=None, cmap="hot", show_fig=False,
                         save_formats="pdf")

        spike_shape = 'o'
        if ms.spike_struct.spike_nums is None:
            continue
        n_cells = len(ms.spike_struct.spike_nums)
        plot_spikes_raster(spike_nums=ms.spike_struct.spike_nums[sorted_indices], param=ms.param,
                           spike_train_format=False,
                           size_fig=(10, 2),
                           file_name=file_name + f"raster_dur__pc_{pc_number}",
                           y_ticks_labels=sorted_indices,
                           y_ticks_labels_size=1,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           without_activity_sum=True,
                           show_sum_spikes_as_percentage=False,
                           span_area_only_on_raster=False,
                           spike_shape=spike_shape,
                           spike_shape_size=0.5,
                           save_formats=["pdf"])

        with open(os.path.join(path_results, file_name + f"_pc_{pc_number}.txt"), "w", encoding='UTF-8') as file:
            file.write(f"Sorted cells" + '\n')
            for i in sorted_indices:
                file.write(f"{i}")
                if i < len(sorted_indices) - 1:
                    file.write(" ")
            file.write(f"" + '\n')