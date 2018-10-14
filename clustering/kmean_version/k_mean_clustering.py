import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
from pattern_discovery.clustering.cluster_tools import detect_cluster_activations_with_sliding_window
from pattern_discovery.display.raster import plot_spikes_raster
from pattern_discovery.display.raster import plot_sum_active_clusters
from pattern_discovery.display.misc import plot_hist_clusters_by_sce
from pattern_discovery.seq_solver.markov_way import order_spike_nums_by_seq



# normalized co-variance
def covnorm(m_sces):
    nb_events = np.shape(m_sces)[1]
    co_var_matrix = np.zeros((nb_events, nb_events))
    for i in np.arange(nb_events):
        for j in np.arange(nb_events):
            if np.correlate(m_sces[:, i], m_sces[:, j]) == 0:
                co_var_matrix[i, j] = 0
            else:
                co_var_matrix[i, j] = np.correlate(m_sces[:, i], m_sces[:, j]) / np.std(m_sces[:, i]) \
                                      / np.std(m_sces[:, j]) / nb_events
    return co_var_matrix


def surrogate_clustering(m_sces, n_clusters, n_surrogate, n_trials, perc_thresholds,
                         fct_to_keep_best_silhouettes, debug_mode=False):
    """

    :param m_sces: sce matrix used for the clustering after permutation
    :param n_clusters: number of clusters
    :param n_surrogate: number of surrogates
    :param n_trials: number of trials by surrogate, keeping one avg silhouette by surrogate
    :param perc_thresholds: list of threshold as percentile
    :param debug_mode:
    :return: a list of value representing the nth percentile over the average threshold of each surrogate, keeping
    each individual silhouette score, not just the mean of each surrogate
    """
    surrogate_silhouettes = np.zeros(n_surrogate * n_clusters)
    m_sces = np.copy(m_sces)
    for j in np.arange(len(m_sces[0])):
        m_sces[:, j] = np.random.permutation(m_sces[:, j])
    for i, s in enumerate(m_sces):
        # print(f"pos before permutation {np.where(s)[0]}")
        m_sces[i] = np.random.permutation(s)

    for surrogate_index in np.arange(n_surrogate):
        best_silhouettes_clusters_avg = None
        best_median_silhouettes = 0
        for trial in np.arange(n_trials):
            # co_var = np.cov(m_sces)
            kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
            cluster_labels = kmeans.labels_
            silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
            sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
            local_clusters_silhouette = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values
                # ith_cluster_silhouette_values.sort()
            med = fct_to_keep_best_silhouettes(local_clusters_silhouette)
            if med > best_median_silhouettes:
                best_median_silhouettes = med
                best_silhouettes_clusters_avg = local_clusters_silhouette
        index = surrogate_index * n_clusters
        surrogate_silhouettes[index:index + n_clusters] = best_silhouettes_clusters_avg
    if debug_mode:
        print(f'end shuffling for {n_clusters} clusters and {n_surrogate} surrogates')
    percentile_results = []
    for perc_threshold in perc_thresholds:
        percentile_results.append(np.percentile(surrogate_silhouettes, perc_threshold))
    return percentile_results


def co_var_first_and_clusters(cells_in_sce, range_n_clusters, fct_to_keep_best_silhouettes=np.median,
                              shuffling=False, n_surrogate=100,
                              nth_best_clusters=-1, neurons_labels=None,
                              plot_matrix=False, data_str="", path_results=None,
                              perc_thresholds_for_surrogate=(95, 99), debug_mode=False):
    """

    :param cells_in_sce:
    :param range_n_clusters:
    :param fct_to_keep_best_silhouettes: function used to keep the best trial, will be applied on all the silhouette
    scores of each trials, the max will be kept
    :param shuffling:
    :param nth_best_clusters: how many clusters to return, if -1 return them all
    :param plot_matrix:
    :param data_str:
    :return:
    """
    # ms = mouse_session
    # param = ms.param
    m_sces = cells_in_sce
    # ax = sns.heatmap(m_sces, cmap="jet")  # , vmin=0, vmax=1) YlGnBu
    # ax.invert_yaxis()
    # plt.show()
    # m_sces = np.transpose(cellsinpeak)
    # m_sces = np.cov(m_sces)
    # print(f'np.shape(m_sces) {np.shape(m_sces)}')
    m_sces = covnorm(m_sces)
    # m_sces = np.corrcoef(m_sces)

    # ax = sns.heatmap(m_sces, cmap="jet")  # , vmin=0, vmax=1) YlGnBu
    # ax.invert_yaxis()
    # plt.show()

    original_m_sces = m_sces
    testing = True

    # key is the nth clusters as int, value is a list of list of SCE
    # (each list representing a cluster, so we have as many list as the number of cluster wanted)
    dict_best_clusters = dict()
    # the key is the nth cluster (int) and the value is a list of cluster number for each cell
    cluster_labels_for_neurons = dict()
    for i in range_n_clusters:
        dict_best_clusters[i] = []

    # nb of time to apply one given number of cluster
    n_trials = 100
    best_kmeans_by_cluster = dict()
    surrogate_percentiles_by_n_cluster = dict()
    for n_clusters in range_n_clusters:
        if debug_mode:
            print(f"n_clusters {n_clusters}")
        surrogate_percentiles = []
        if shuffling:
            surrogate_percentiles = surrogate_clustering(m_sces=m_sces, n_clusters=n_clusters,
                                                         n_surrogate=n_surrogate,
                                                         n_trials=100,
                                                         fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                                                         perc_thresholds=perc_thresholds_for_surrogate, debug_mode=True)
            if debug_mode:
                print(f"surrogate_percentiles {surrogate_percentiles}")

        best_kmeans = None
        silhouette_avgs = np.zeros(n_trials)
        best_silhouettes_clusters_avg = None
        max_local_clusters_silhouette = 0
        best_silhouettes = 0
        silhouettes_clusters_avg = []
        for trial in np.arange(n_trials):
            # co_var = np.cov(m_sces)
            kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
            cluster_labels = kmeans.labels_
            # if np.max(cluster_labels) == 0:
            #     print(f"only one cluster for {n_clusters} clusters")
            #     break
            silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
            silhouette_avgs[trial] = silhouette_avg
            # print(f"Avg silhouette: {silhouette_avg}")
            sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
            local_clusters_silhouette = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                # print(f'ith_cluster_silhouette_values {ith_cluster_silhouette_values}')
                # print(f'np.mean(ith_cluster_silhouette_values) {np.mean(ith_cluster_silhouette_values)}')
                avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                silhouettes_clusters_avg.append(avg_ith_cluster_silhouette_values)
                # print(ith_cluster_silhouette_values)
                local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values
                # ith_cluster_silhouette_values.sort()
            # compute a score based on the silhouette of each cluster for this trial and compare it with the best score
            # so far, keeping it if it's better
            computed_score = fct_to_keep_best_silhouettes(local_clusters_silhouette)
            if computed_score > best_silhouettes:
                best_silhouettes = computed_score
                best_silhouettes_clusters_avg = local_clusters_silhouette

            max_local = computed_score  # np.max(local_clusters_silhouette)  # silhouette_avg
            # TO display, we keep the group with the cluster with the max silhouette
            if (best_kmeans is None) or (computed_score > max_local_clusters_silhouette):
                max_local_clusters_silhouette = max_local
                best_kmeans = kmeans
                nth_best_list = []
                count_clusters = nth_best_clusters
                if count_clusters == -1:
                    count_clusters = n_clusters
                for b in np.arange(count_clusters):
                    arg = np.argmax(local_clusters_silhouette)
                    # TODO: put neurons list instead of SCEs
                    nth_best_list.append(np.arange(len(m_sces))[cluster_labels == arg])
                    local_clusters_silhouette[arg] = -1
                dict_best_clusters[n_clusters] = nth_best_list
            # silhouettes_clusters_avg.extend(sample_silhouette_values)

            # if best_kmeans is None:
            #     continue
        best_kmeans_by_cluster[n_clusters] = best_kmeans
        cluster_labels_for_neurons[n_clusters] = \
            find_cluster_labels_for_neurons(cells_in_peak=cells_in_sce,
                                            cluster_labels=best_kmeans.labels_)
        surrogate_percentiles_by_n_cluster[n_clusters] = surrogate_percentiles
        if plot_matrix:
            show_co_var_first_matrix(cells_in_peak=np.copy(cells_in_sce), m_sces=m_sces,
                                     n_clusters=n_clusters, kmeans=best_kmeans,
                                     cluster_labels_for_neurons=cluster_labels_for_neurons[n_clusters],
                                     data_str=data_str, path_results=path_results,
                                     show_silhouettes=True, neurons_labels=neurons_labels,
                                     surrogate_silhouette_avg=surrogate_percentiles)

    return dict_best_clusters, best_kmeans_by_cluster, m_sces, cluster_labels_for_neurons, \
           surrogate_percentiles_by_n_cluster


# TODO: do shuffling before the real cluster
# TODO: when showing co-var, show the 95th percentile of shuffling, to see which cluster is significant

def show_co_var_first_matrix(cells_in_peak, m_sces, n_clusters, kmeans, cluster_labels_for_neurons,
                             data_str, path_results=None, show_fig=False, show_silhouettes=False,
                             surrogate_silhouette_avg=None, neurons_labels=None, axes_list=None, fig_to_use=None,
                             save_formats="pdf"):
    n_cells = len(cells_in_peak)

    if axes_list is None:
        if show_silhouettes:
            fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharex=False,
                                                gridspec_kw={'height_ratios': [1], 'width_ratios': [6, 6, 10]},
                                                figsize=(20, 12))
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=False,
                                           gridspec_kw={'height_ratios': [1], 'width_ratios': [4, 10]},
                                           figsize=(20, 12))
        plt.tight_layout(pad=3, w_pad=7, h_pad=3)
        # ax1 = plt.subplot(121)
        plt.title(f"{data_str} {n_clusters} clusters")
    else:
        if show_silhouettes:
            ax0, ax1, ax2 = axes_list
        else:
            ax1, ax2 = axes_list
    # list of size nb_sce, each sce having a value from 0 to k clusters
    cluster_labels = kmeans.labels_

    if show_silhouettes:
        # Compute the silhouette scores for each sample
        sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels)
        ax0.set_facecolor("black")
        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i + 1) / (n_clusters + 1))
            ax0.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax0.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color="white")

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax0.set_title("The silhouette plot for the various clusters.")
        ax0.set_xlabel("The silhouette coefficient values")
        ax0.set_ylabel("Cluster label")

        silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
        if surrogate_silhouette_avg is not None:
            for value in surrogate_silhouette_avg:
                ax0.axvline(x=value, color="white", linestyle="--")
        # The vertical line for average silhouette score of all the values
        ax0.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax0.set_yticks([])  # Clear the yaxis labels / ticks
        ax0.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # contains the neurons from the SCE, but ordered by cluster
    # print(f'// np.shape(m_sces) {np.shape(m_sces)}')
    ordered_m_sces = np.zeros((np.shape(m_sces)[0], np.shape(m_sces)[1]))
    # to plot line that separate clusters
    cluster_coord_thresholds = []
    cluster_x_ticks_coord = []
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_m_sces[start:start + nb_k, :] = m_sces[e, :]
        ordered_m_sces[:, start:start + nb_k] = m_sces[:, e]
        start += nb_k
        if (k + 1) < n_clusters:
            if k == 0:
                cluster_x_ticks_coord.append(start / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_coord_thresholds[-1]) / 2)
            cluster_coord_thresholds.append(start)
        else:
            cluster_x_ticks_coord.append((start + cluster_coord_thresholds[-1]) / 2)

    co_var = np.corrcoef(ordered_m_sces)  # cov
    # sns.set()
    result = sns.heatmap(co_var, cmap="Blues", ax=ax1)  # , vmin=0, vmax=1) YlGnBu  cmap="jet" Blues
    # ax1.hlines(cluster_coord_thresholds, 0, np.shape(co_var)[0], color="black", linewidth=1,
    #            linestyles="dashed")
    for n_c, clusters_threshold in enumerate(cluster_coord_thresholds):
        # if (n_c+1) == len(cluster_coord_thresholds):
        #     break
        x_begin = 0
        if n_c > 0:
            x_begin = cluster_coord_thresholds[n_c - 1]
        x_end = np.shape(co_var)[0]
        if n_c < len(cluster_coord_thresholds) - 1:
            x_end = cluster_coord_thresholds[n_c + 1]
        ax1.hlines(clusters_threshold, x_begin, x_end, color="black", linewidth=2,
                   linestyles="dashed")
    for n_c, clusters_threshold in enumerate(cluster_coord_thresholds):
        # if (n_c+1) == len(cluster_coord_thresholds):
        #     break
        y_begin = 0
        if n_c > 0:
            y_begin = cluster_coord_thresholds[n_c - 1]
        y_end = np.shape(co_var)[0]
        if n_c < len(cluster_coord_thresholds) - 1:
            y_end = cluster_coord_thresholds[n_c + 1]
        ax1.vlines(clusters_threshold, y_begin, y_end, color="black", linewidth=2,
                   linestyles="dashed")
    # ax1.xaxis.get_majorticklabels().set_rotation(90)
    # plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    # plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    ax1.set_xticks(cluster_x_ticks_coord)
    ax1.set_xticklabels(np.arange(n_clusters))
    ax1.set_yticks(cluster_x_ticks_coord)
    ax1.set_yticklabels(np.arange(n_clusters))
    ax1.set_title(f"{np.shape(m_sces)[0]} SCEs")
    # ax1.xaxis.set_tick_params(labelsize=5)
    # ax1.yaxis.set_tick_params(labelsize=5)
    ax1.invert_yaxis()

    # plt.show()
    original_cells_in_peak = cells_in_peak

    cells_in_peak = np.copy(original_cells_in_peak)
    # ax2 = plt.subplot(1, 2, 2)

    # first order assemblies
    ordered_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]), dtype="int16")
    # then order neurons
    ordered_n_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]), dtype="int16")
    cluster_vertical_thresholds = []
    cluster_x_ticks_coord = []
    cluster_horizontal_thresholds = []
    # key is the cluster number and value is a tuple of int
    clusters_coords_dict = dict()
    cells_cluster_dict = dict()
    # set the number of neurons for whom there are no spikes or less than 2 for a given cluster
    nb_neurons_without_clusters = 0
    # if True, will put spike in each cluster in the color of the cluster, by putting the matrix value to the value of
    # the cluster, if False, will set in a darker color the cluster that belong to a cell
    color_each_clusters = True
    neurons_normal_order = np.arange(np.shape(cells_in_peak)[0])
    neurons_ax_labels = np.zeros(np.shape(cells_in_peak)[0], dtype="int16")
    # key is the cluster number, k, and value is an np.array of int reprenseting the indices of SCE part of this cluster
    sce_indices_for_each_clusters = dict()
    new_sce_labels = np.zeros(np.shape(cells_in_peak)[1], dtype="int16")
    nb_cells_by_cluster_of_cells = []
    nb_cells_by_cluster_of_cells_y_coord = []
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        if color_each_clusters:
            for cell in np.arange(len(cells_in_peak)):
                spikes_index = np.where(cells_in_peak[cell, e] == 1)[0]
                to_put_to_true = np.where(e)[0][spikes_index]
                tmp_e = np.zeros(len(e), dtype="bool")
                tmp_e[to_put_to_true] = True
                # print(f"cell {cell}, spikes_index {spikes_index}, "
                #       f"cells_in_peak[cell, :] {cells_in_peak[cell, :]},"
                #       f" cells_in_peak[cell, spikes_index] {cells_in_peak[cell, spikes_index]}")
                # K +2 to avoid zero and one, and at the end we will substract 1
                # print(f"cell {cell}, k {k}, cells_in_peak[cell, :] {cells_in_peak[cell, :]}")
                cells_in_peak[cell, tmp_e] = k + 2
                # print(f"cells_in_peak[cell, :] {cells_in_peak[cell, :]}")

        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        sce_indices_for_each_clusters[k] = np.arange(start, start + nb_k)
        old_pos = np.where(e)[0]
        for i, sce_index in enumerate(np.arange(start, start + nb_k)):
            new_sce_labels[sce_index] = old_pos[i]
        start += nb_k
        if (k + 1) < n_clusters:
            if k == 0:
                cluster_x_ticks_coord.append(start / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_vertical_thresholds[-1]) / 2)
            cluster_vertical_thresholds.append(start)
        else:
            cluster_x_ticks_coord.append((start + cluster_vertical_thresholds[-1]) / 2)

    start = 0

    for k in np.arange(-1, np.max(cluster_labels_for_neurons) + 1):
        e = np.equal(cluster_labels_for_neurons, k)
        nb_cells = np.sum(e)
        if nb_cells == 0:
            # print(f'n_clusters {n_clusters}, k {k} nb_cells == 0')
            continue
        # print(f'nb_k {nb_k}, k: {k}')
        if k == -1:
            nb_neurons_without_clusters = nb_cells
        else:
            if not color_each_clusters:
                sce_indices = np.array(sce_indices_for_each_clusters[k])
                # print(f"sce_indices {sce_indices}, np.shape(ordered_cells_in_peak) {np.shape(ordered_cells_in_peak)}, "
                #       f"e {e} ")
                # we put to a value > 1 the sce where the neuron has a spike in their assigned cluster
                # to_modify = ordered_cells_in_peak[e, :][:, mask]
                for index in sce_indices:
                    tmp_e = np.copy(e)
                    # keeping for each sce, the cells that belong to cluster k
                    tmp_array = ordered_cells_in_peak[tmp_e, index]
                    # finding which cells don't have spikes
                    pos = np.where(tmp_array == 0)[0]
                    to_put_to_false = np.where(tmp_e)[0][pos]
                    tmp_e[to_put_to_false] = False
                    # putting to 2 all cells for whom there is a spike
                    ordered_cells_in_peak[tmp_e, index] = 2
                # to_modify[np.where(to_modify)[0]] = 2
        ordered_n_cells_in_peak[start:start + nb_cells, :] = ordered_cells_in_peak[e, :]
        neurons_ax_labels[start:start + nb_cells] = neurons_normal_order[e]
        nb_cells_by_cluster_of_cells.append(nb_cells)
        nb_cells_by_cluster_of_cells_y_coord.append(start + (nb_cells / 2))
        for cell in np.arange(start, start + nb_cells):
            cells_cluster_dict[cell] = k
        clusters_coords_dict[k] = (start, start + nb_cells)
        start += nb_cells
        if (k + 1) < (np.max(cluster_labels_for_neurons) + 1):
            cluster_horizontal_thresholds.append(start)

    if color_each_clusters:
        # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
        ordered_n_cells_in_peak = ordered_n_cells_in_peak - 2
        # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
        list_color = ['black']
        bounds = [-2.5, -0.5]
        for i in np.arange(n_clusters):
            color = cm.nipy_spectral(float(i + 1) / (n_clusters + 1))
            list_color.append(color)
            bounds.append(i + 0.5)
        cmap = mpl.colors.ListedColormap(list_color)
    else:
        # light_blue_color = [0, 0.871, 0.219]
        cmap = mpl.colors.ListedColormap(['black', 'cornflowerblue', 'blue'])
        # cmap.set_over('red')
        # cmap.set_under('blue')
        bounds = [-0.5, 0.5, 1.5, 2.5]

    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    sns.heatmap(ordered_n_cells_in_peak, cbar=False, ax=ax2, cmap=cmap, norm=norm)
    # print(f"len(neurons_ax_labels) {len(neurons_ax_labels)}")
    # TODO: set the position of labels, right now only one on two are displayed, fontsize should be decreased
    if neurons_labels is not None:
        ordered_neurons_labels = []
        for index in neurons_ax_labels:
            ordered_neurons_labels.append(neurons_labels[index])
        ax2.set_yticks(np.arange(len(ordered_neurons_labels)) + 0.5)
        ax2.set_yticklabels(ordered_neurons_labels)

    else:
        ax2.set_yticks(np.arange(len(neurons_ax_labels)))
        ax2.set_yticklabels(neurons_ax_labels.astype(int))

    if len(neurons_ax_labels) > 100:
        ax2.yaxis.set_tick_params(labelsize=4)
    elif len(neurons_ax_labels) > 200:
        ax2.yaxis.set_tick_params(labelsize=3)
    elif len(neurons_ax_labels) > 400:
        ax2.yaxis.set_tick_params(labelsize=2)
    else:
        ax2.yaxis.set_tick_params(labelsize=8)

    # creating axis at the top
    ax_top = ax2.twiny()
    ax_right = ax2.twinx()
    ax2.set_frame_on(False)
    ax_top.set_frame_on(False)
    ax_top.set_xlim((0, np.shape(cells_in_peak)[1]))
    ax_top.set_xticks(cluster_x_ticks_coord)
    # clusters labels
    ax_top.set_xticklabels(np.arange(n_clusters))

    # print(f"nb_cells_by_cluster_of_cells_y_coord {nb_cells_by_cluster_of_cells_y_coord} "
    #       f"nb_cells_by_cluster_of_cells {nb_cells_by_cluster_of_cells}")
    ax_right.set_frame_on(False)
    ax_right.set_ylim((0, len(neurons_ax_labels)))
    ax_right.set_yticks(nb_cells_by_cluster_of_cells_y_coord)
    # clusters labels
    ax_right.set_yticklabels(nb_cells_by_cluster_of_cells)

    ax2.set_xticks(np.arange(np.shape(cells_in_peak)[1]) + 0.5)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    # sce labels
    ax2.set_xticklabels(new_sce_labels)
    if len(new_sce_labels) > 100:
        ax2.xaxis.set_tick_params(labelsize=4)
    elif len(new_sce_labels) > 200:
        ax2.xaxis.set_tick_params(labelsize=3)
    elif len(new_sce_labels) > 400:
        ax2.xaxis.set_tick_params(labelsize=2)
    else:
        ax2.xaxis.set_tick_params(labelsize=6)
    ax2.hlines(cluster_horizontal_thresholds, 0, np.shape(cells_in_peak)[1], color="red", linewidth=1,
               linestyles="dashed")
    ax2.vlines(cluster_vertical_thresholds, 0, np.shape(cells_in_peak)[0], color="red", linewidth=1,
               linestyles="dashed")
    # print(f"n_clusters {n_clusters}, cluster_vertical_thresholds {cluster_vertical_thresholds}")
    for cluster in np.arange(n_clusters):
        if cluster not in clusters_coords_dict:
            # print(f"cluster {cluster} with no cells")
            # means no cell has this cluster as main cluster
            continue
        y_bottom, y_top = clusters_coords_dict[cluster]
        x_left = 0 if (cluster == 0) else cluster_vertical_thresholds[cluster - 1]
        x_right = np.shape(cells_in_peak)[1] if (cluster == (n_clusters - 1)) else cluster_vertical_thresholds[cluster]
        linewidth = 3
        color_border = "white"
        ax2.vlines(x_left, y_bottom, y_top, color=color_border, linewidth=linewidth)
        ax2.vlines(x_right, y_bottom, y_top, color=color_border, linewidth=linewidth)
        ax2.hlines(y_bottom, x_left, x_right, color=color_border, linewidth=linewidth)
        ax2.hlines(y_top, x_left, x_right, color=color_border, linewidth=linewidth)

    # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)

    for cell in np.arange(n_cells):
        cluster = cells_cluster_dict[cell]
        if cluster >= 0:
            color = cm.nipy_spectral(float(cluster + 1) / (n_clusters + 1))
            ax2.get_yticklabels()[cell].set_color(color)

    ax2.invert_yaxis()
    # if nb_neurons_without_clusters > 0:
    #     for i in np.arange(nb_neurons_without_clusters):
    #         ax2.get_yticklabels()[i].set_color("red")

    if (path_results is not None) and ((axes_list is None) or (fig_to_use is not None)):
        if fig_to_use is not None:
            fig = fig_to_use
        fig.savefig(f'{path_results}/{n_clusters}_clusters_{data_str}.{save_formats}',
                    format=f"{save_formats}")
    if show_fig:
        plt.show()
        plt.close()


def find_cluster_labels_for_neurons(cells_in_peak, cluster_labels):
    cluster_labels_for_neurons = np.zeros(np.shape(cells_in_peak)[0], dtype="int8")
    # sorting neurons spikes, keeping them only in one cluster, the one with the most spikes from this neuron
    # if spikes < 2 in any clusters, then removing spikes
    # going neuron by neuron,
    # removing_multiple_spikes_among_cluster = False

    for n, events in enumerate(cells_in_peak):
        pos_events = np.where(events)[0]
        max_clusters = np.zeros(np.max(cluster_labels) + 1, dtype="int8")
        for p in pos_events:
            # p correspond to the index of one SCE
            max_clusters[cluster_labels[p]] += 1
        if np.max(max_clusters) < 2:
            # if removing_multiple_spikes_among_cluster:
            #     cells_in_peak[n, :] = np.zeros(len(cells_in_peak[n, :]))
            cluster_labels_for_neurons[n] = -1
        else:
            # selecting the cluster with the most spikes from neuron n
            max_cluster = np.argmax(max_clusters)
            cluster_labels_for_neurons[n] = max_cluster
            # clearing spikes from other cluster
            # if removing_multiple_spikes_among_cluster:
            #     cells_in_peak[n, np.not_equal(cluster_labels, max_cluster)] = 0
    return cluster_labels_for_neurons


def save_stat_SCE_and_cluster_k_mean_version(spike_nums_to_use, activity_threshold, k_means,
                                             SCE_times, n_cluster, param, sliding_window_duration,
                                             cluster_labels_for_neurons, perc_threshold,
                                             n_surrogate_k_mean, data_descr,
                                             n_surrogate_activity_threshold):
    round_factor = 2
    file_name = f'{param.path_results}/stat_k_mean_v_{data_descr}_{n_cluster}_clusters_{param.time_str}.txt'
    with open(file_name, "w", encoding='UTF-8') as file:
        file.write(f"Stat k_mean version for {n_cluster} clusters" + '\n')
        file.write("" + '\n')
        file.write(f"cells {len(spike_nums_to_use)}, events {len(SCE_times)}" + '\n')
        file.write(f"Event participation threshold {activity_threshold}, {perc_threshold} percentile, "
                   f"{n_surrogate_activity_threshold} surrogates" + '\n')
        file.write(f"Sliding window duration {sliding_window_duration}" + '\n')
        file.write(f"{n_surrogate_k_mean} surrogates for kmean" + '\n')
        file.write("" + '\n')
        file.write("" + '\n')
        cluster_labels = k_means.labels_

        for k in np.arange(n_cluster):

            e = np.equal(cluster_labels, k)

            nb_sce_in_cluster = np.sum(e)
            sce_ids = np.where(e)[0]

            e_cells = np.equal(cluster_labels_for_neurons, k)
            n_cells_in_cluster = np.sum(e_cells)

            file.write("#" * 10 + f"   cluster {k} / {nb_sce_in_cluster} events / {n_cells_in_cluster} cells" +
                       "#" * 10 + '\n')
            file.write('\n')

            duration_values = np.zeros(nb_sce_in_cluster, dtype="uint16")
            max_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")
            mean_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")
            overall_activity_values = np.zeros(nb_sce_in_cluster, dtype="float")

            for n, sce_id in enumerate(sce_ids):
                duration_values[n], max_activity_values[n], \
                mean_activity_values[n], overall_activity_values[n] = \
                    give_stat_one_sce(sce_id=sce_id,
                                      spike_nums_to_use=spike_nums_to_use,
                                      SCE_times=SCE_times, sliding_window_duration=sliding_window_duration)
            file.write(f"Duration: mean {np.round(np.mean(duration_values), round_factor)}, "
                       f"std {np.round(np.std(duration_values), round_factor)}, "
                       f"median {np.round(np.median(duration_values), round_factor)}\n")
            file.write(f"Overall participation: mean {np.round(np.mean(overall_activity_values), round_factor)}, "
                       f"std {np.round(np.std(overall_activity_values), round_factor)}, "
                       f"median {np.round(np.median(overall_activity_values), round_factor)}\n")
            file.write(f"Max participation: mean {np.round(np.mean(max_activity_values), round_factor)}, "
                       f"std {np.round(np.std(max_activity_values), round_factor)}, "
                       f"median {np.round(np.median(max_activity_values), round_factor)}\n")
            file.write(f"Mean participation: mean {np.round(np.mean(mean_activity_values), round_factor)}, "
                       f"std {np.round(np.std(mean_activity_values), round_factor)}, "
                       f"median {np.round(np.median(mean_activity_values), round_factor)}\n")
            file.write('\n')

        file.write('\n')
        file.write('\n')
        file.write("#" * 50 + '\n')
        file.write('\n')
        file.write('\n')
        # for each SCE
        for sce_id in np.arange(len(SCE_times)):
            result = give_stat_one_sce(sce_id=sce_id,
                                       spike_nums_to_use=spike_nums_to_use,
                                       SCE_times=SCE_times,
                                       sliding_window_duration=sliding_window_duration)
            duration_in_frames, max_activity, mean_activity, overall_activity = result
            file.write(f"SCE {sce_id}" + '\n')
            file.write(f"Duration_in_frames {duration_in_frames}" + '\n')
            file.write(f"Overall participation {np.round(overall_activity, round_factor)}" + '\n')
            file.write(f"Max participation {np.round(max_activity, round_factor)}" + '\n')
            file.write(f"Mean participation {np.round(mean_activity, round_factor)}" + '\n')

            file.write('\n')
            file.write('\n')


def give_stat_one_sce(sce_id, spike_nums_to_use, SCE_times, sliding_window_duration):
    """

    :param sce_id:
    :param spike_nums_to_use:
    :param SCE_times:
    :param sliding_window_duration:
    :return: duration_in_frames: duration of the sce in frames
     max_activity: the max number of cells particpating during a window_duration to the sce
     mean_activity: the mean of cells particpating during the sum of window_duration
     overall_activity: the number of different cells participating to the SCE all along
     if duration == sliding_window duration, max_activity, mean_activity and overall_activity will be equal
    """
    time_tuple = SCE_times[sce_id]
    duration_in_frames = (time_tuple[1] - time_tuple[0]) + 1
    n_slidings = (duration_in_frames - sliding_window_duration) + 1
    # print(f"duration_in_frames {duration_in_frames}")
    # print(f"sliding_window_duration {sliding_window_duration}")
    # print(f"time_tuple[1] {time_tuple[1]}, time_tuple[0] {time_tuple[0]}")
    # print(f"n_slidings {n_slidings}")
    sum_activity_for_each_frame = np.zeros(n_slidings)
    for n in np.arange(n_slidings):
        # see to use window_duration to find the amount of participation
        time_beg = time_tuple[0] + n
        sum_activity_for_each_frame[n] = len(np.where(np.sum(spike_nums_to_use[:,
                                                             time_beg:(time_beg + sliding_window_duration)],
                                                             axis=1))[0])
    max_activity = np.max(sum_activity_for_each_frame)
    mean_activity = np.mean(sum_activity_for_each_frame)
    overall_activity = len(np.where(np.sum(spike_nums_to_use[:,
                                           time_tuple[0]:(time_tuple[1] + 1)], axis=1))[0])

    return duration_in_frames, max_activity, mean_activity, overall_activity


def compute_and_plot_clusters_raster_kmean_version(labels, activity_threshold, range_n_clusters_k_mean,
                                                  n_surrogate_k_mean,
                                                  spike_nums_to_use, cellsinpeak, data_descr,
                                                  param,
                                                  sliding_window_duration, sce_times_numbers,
                                                  SCE_times, perc_threshold,
                                                   n_surrogate_activity_threshold,
                                                   with_shuffling=True,
                                                   debug_mode=False,
                                                   with_cells_in_cluster_seq_sorted=False):
    # perc_threshold is the number of percentile choosen to determine the threshold
    #
    # -------- clustering params ------ -----
    # range_n_clusters_k_mean = np.arange(2, 17)
    # n_surrogate_k_mean = 100


    clusters_sce, best_kmeans_by_cluster, m_cov_sces, cluster_labels_for_neurons, surrogate_percentiles = \
        co_var_first_and_clusters(cells_in_sce=cellsinpeak, shuffling=with_shuffling,
                                  n_surrogate=n_surrogate_k_mean,
                                  fct_to_keep_best_silhouettes=np.mean,
                                  range_n_clusters=range_n_clusters_k_mean,
                                  nth_best_clusters=-1,
                                  plot_matrix=False,
                                  data_str=data_descr,
                                  path_results=param.path_results,
                                  neurons_labels=labels,
                                  debug_mode=debug_mode)

    if with_cells_in_cluster_seq_sorted:
        data_descr = data_descr + "_seq"

    for n_cluster in range_n_clusters_k_mean:
        clustered_spike_nums = np.copy(spike_nums_to_use)
        cell_labels = []
        cluster_labels = cluster_labels_for_neurons[n_cluster]
        cluster_horizontal_thresholds = []
        cells_to_highlight = []
        cells_to_highlight_colors = []
        start = 0
        for k in np.arange(-1, np.max(cluster_labels) + 1):
            e = np.equal(cluster_labels, k)
            nb_k = np.sum(e)
            if nb_k==0:
                continue
            cells_indices = np.where(e)[0]
            if with_cells_in_cluster_seq_sorted and (len(cells_indices) > 2):
                result_ordering = order_spike_nums_by_seq(spike_nums_to_use[cells_indices, :], param,
                                                                        debug_mode=debug_mode)
                seq_dict_tmp, ordered_indices, all_best_seq = result_ordering
                # if a list of ordered_indices, the size of the list is equals to ne number of cells,
                # each list correspond to the best order with this cell as the first one in the ordered seq
                if ordered_indices is not None:
                    cells_indices = cells_indices[ordered_indices]
            clustered_spike_nums[start:start + nb_k, :] = spike_nums_to_use[cells_indices, :]
            for index in cells_indices:
                cell_labels.append(labels[index])
            if k >= 0:
                color = cm.nipy_spectral(float(k + 1) / (n_cluster + 1))
                cell_indices = list(np.arange(start, start + nb_k))
                cells_to_highlight.extend(cell_indices)
                cells_to_highlight_colors.extend([color] * len(cell_indices))
            start += nb_k
            if (k + 1) < (np.max(cluster_labels) + 1):
                cluster_horizontal_thresholds.append(start)

        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 2})
        outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

        inner_top = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                     subplot_spec=outer[1], height_ratios=[10, 2])

        # clusters display
        inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 3,
                                                        subplot_spec=outer[0], width_ratios=[6, 10, 6])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_top[0])
        # ax2 contains the peak activity diagram
        ax2 = fig.add_subplot(inner_top[1], sharex=ax1)

        ax3 = fig.add_subplot(inner_bottom[0])
        # ax2 contains the peak activity diagram
        ax4 = fig.add_subplot(inner_bottom[1])
        ax5 = fig.add_subplot(inner_bottom[2])
        if len(cell_labels) > 100:
            y_ticks_labels_size = 1
        else:
            y_ticks_labels_size = 3
        plot_spikes_raster(spike_nums=clustered_spike_nums, param=param,
                           spike_train_format=False,
                           title=f"{n_cluster} clusters raster plot {data_descr}",
                           file_name=f"spike_nums_{data_descr}_{n_cluster}_clusters",
                           y_ticks_labels=cell_labels,
                           y_ticks_labels_size=y_ticks_labels_size,
                           save_raster=False,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=activity_threshold,
                           span_cells_to_highlight=False,
                           raster_face_color='black',
                           cell_spikes_color='white',
                           horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                           horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                           horizontal_lines_sytle="dashed",
                           horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                           vertical_lines=SCE_times,
                           vertical_lines_colors=['white'] * len(SCE_times),
                           vertical_lines_sytle="solid",
                           vertical_lines_linewidth=[0.2] * len(SCE_times),
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="pdf",
                           axes_list=[ax1, ax2],
                           SCE_times=SCE_times)

        show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
                                 n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                 data_str=data_descr, path_results=param.path_results,
                                 show_silhouettes=True, neurons_labels=labels,
                                 surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
                                 axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats="pdf")
        plt.close()

        # ######### Plot that show cluster activation

        result_detection = detect_cluster_activations_with_sliding_window(spike_nums=spike_nums_to_use,
                                                                          window_duration=sliding_window_duration,
                                                                          cluster_labels=cluster_labels,
                                                                          sce_times_numbers=sce_times_numbers,
                                                                          debug_mode=False)

        clusters_activations_by_cell, clusters_activations_by_cluster, cluster_particpation_to_sce, \
        clusters_corresponding_index = result_detection
        # print(f"cluster_particpation_to_sce {cluster_particpation_to_sce}")

        cell_labels = []
        cluster_horizontal_thresholds = []
        cells_to_highlight = []
        cells_to_highlight_colors = []
        start = 0
        for k in np.arange(np.max(cluster_labels) + 1):
            e = np.equal(cluster_labels, k)
            nb_k = np.sum(e)
            if nb_k == 0:
                continue
            for index in np.where(e)[0]:
                cell_labels.append(labels[index])
            if k >= 0:
                color = cm.nipy_spectral(float(k + 1) / (n_cluster + 1))
                cell_indices = list(np.arange(start, start + nb_k))
                cells_to_highlight.extend(cell_indices)
                cells_to_highlight_colors.extend([color] * len(cell_indices))
            start += nb_k
            if (k + 1) < (np.max(cluster_labels) + 1):
                cluster_horizontal_thresholds.append(start)

        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 3})
        outer = gridspec.GridSpec(1, 1)  # , height_ratios=[60, 40])

        # clusters display
        # inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
        #                                              subplot_spec=outer[0])

        inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                        subplot_spec=outer[0], height_ratios=[10, 2])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_bottom[0])
        # ax3 contains the peak activity diagram
        ax2 = fig.add_subplot(inner_bottom[1], sharex=ax1)

        plot_spikes_raster(spike_nums=clusters_activations_by_cell, param=param,
                           spike_train_format=False,
                           file_name=f"raster_clusters_detection_{data_descr}",
                           y_ticks_labels=cell_labels,
                           y_ticks_labels_size=4,
                           save_raster=True,
                           show_raster=False,
                           plot_with_amplitude=False,
                           span_cells_to_highlight=False,
                           raster_face_color='black',
                           cell_spikes_color='white',
                           horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                           horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                           horizontal_lines_sytle="dashed",
                           vertical_lines=SCE_times,
                           vertical_lines_colors=['white'] * len(SCE_times),
                           vertical_lines_sytle="dashed",
                           vertical_lines_linewidth=[0.4] * len(SCE_times),
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="|",
                           spike_shape_size=1,
                           save_formats="pdf",
                           axes_list=[ax1],
                           without_activity_sum=True,
                           ylabel="")

        # print(f"n_cluster {n_cluster} len(clusters_activations) {len(clusters_activations)}")

        plot_sum_active_clusters(clusters_activations=clusters_activations_by_cluster, param=param,
                                 sliding_window_duration=sliding_window_duration,
                                 data_str=f"raster_{n_cluster}_clusters_participation_{data_descr}",
                                 axes_list=[ax2],
                                 fig_to_use=fig)

        plot_hist_clusters_by_sce(cluster_particpation_to_sce,
                                  data_str=f"hist_percentage_of_network_events_{n_cluster}_clusters",
                                  param=param)

        plt.close()

        save_stat_SCE_and_cluster_k_mean_version(spike_nums_to_use=spike_nums_to_use,
                                                 data_descr=data_descr,
                                                 activity_threshold=activity_threshold,
                                                 k_means=best_kmeans_by_cluster[n_cluster],
                                                 SCE_times=SCE_times, n_cluster=n_cluster, param=param,
                                                 sliding_window_duration=sliding_window_duration,
                                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                                 perc_threshold=perc_threshold,
                                                 n_surrogate_k_mean=n_surrogate_k_mean,
                                                 n_surrogate_activity_threshold=n_surrogate_activity_threshold
                                                 )
