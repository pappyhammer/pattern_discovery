import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm


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


def surrogate_clustering(m_sces, n_clusters, n_surrogate, n_trials, perc_thresholds, debug_mode=False):
    """

    :param m_sces: sce matrix used for the clustering after permutation
    :param n_clusters: number of clusters
    :param n_surrogate: number of surrogates
    :param n_trials: number of trials by surrogate, keeping one avg silhouette by surrogate
    :param perc_thresholds: list of threshold as percentile
    :param debug_mode:
    :return: a list of value representing the nth percentile over the average threshold of each surrogate
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
            med = np.median(local_clusters_silhouette)
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
                              perc_thresholds_for_surrogate=(95, 99)):
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
        print(f"n_clusters {n_clusters}")
        surrogate_percentiles = []
        if shuffling:
            surrogate_percentiles = surrogate_clustering(m_sces=m_sces, n_clusters=n_clusters,
                                                         n_surrogate=n_surrogate,
                                                         n_trials=100,
                                                         perc_thresholds=perc_thresholds_for_surrogate, debug_mode=True)
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
    # list of size nb_neurons, each neuron having a value from 0 to k clusters
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

            color = cm.nipy_spectral(float(i+1) / (n_clusters+1))
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
                ax0.axvline(x=value, color="black", linestyle="--")
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
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        if color_each_clusters:
            for cell in np.arange(len(cells_in_peak)):
                spikes_index = np.where(cells_in_peak[cell, e]==1)[0]
                to_put_to_true = np.where(e)[0][spikes_index]
                tmp_e = np.zeros(len(e), dtype="bool")
                tmp_e[to_put_to_true] = True
                # print(f"cell {cell}, spikes_index {spikes_index}, "
                #       f"cells_in_peak[cell, :] {cells_in_peak[cell, :]},"
                #       f" cells_in_peak[cell, spikes_index] {cells_in_peak[cell, spikes_index]}")
                # K +2 to avoid zero and one, and at the end we will substract 1
                # print(f"cell {cell}, k {k}, cells_in_peak[cell, :] {cells_in_peak[cell, :]}")
                cells_in_peak[cell, tmp_e] = k+2
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
            color = cm.nipy_spectral(float(i+1) / (n_clusters+1))
            list_color.append(color)
            bounds.append(i+0.5)
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
        ax2.yaxis.set_tick_params(labelsize=8)
    else:
        ax2.set_yticks(np.arange(len(neurons_ax_labels)))
        ax2.set_yticklabels(neurons_ax_labels.astype(int))

    # creating axis at the top
    ax_top = ax2.twiny()
    ax2.set_frame_on(False)
    ax_top.set_frame_on(False)
    ax_top.set_xlim((0, np.shape(cells_in_peak)[1]))
    ax_top.set_xticks(cluster_x_ticks_coord)
    # clusters labels
    ax_top.set_xticklabels(np.arange(n_clusters))

    ax2.set_xticks(np.arange(np.shape(cells_in_peak)[1]) + 0.5)
    # sce labels
    ax2.set_xticklabels(new_sce_labels)
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
        x_left = 0 if (cluster == 0) else cluster_vertical_thresholds[cluster-1]
        x_right = np.shape(cells_in_peak)[1] if (cluster == (n_clusters-1)) else cluster_vertical_thresholds[cluster]
        linewidth = 3
        color_border = "white"
        ax2.vlines(x_left, y_bottom, y_top, color = color_border, linewidth = linewidth)
        ax2.vlines(x_right, y_bottom, y_top, color = color_border, linewidth=linewidth)
        ax2.hlines(y_bottom, x_left, x_right, color = color_border, linewidth=linewidth)
        ax2.hlines(y_top, x_left, x_right, color = color_border, linewidth=linewidth)

    # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)
    
    for cell in np.arange(n_cells):
        cluster = cells_cluster_dict[cell]
        if cluster >=0:
            color = cm.nipy_spectral(float(cluster + 1) / (n_clusters+1))
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
