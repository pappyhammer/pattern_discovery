import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.cluster import KMeans
import numpy as np

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


def co_var_first_and_clusters(cells_in_sce, range_n_clusters, shuffling=False, nth_best_clusters=-1,
                              plot_matrix=False, data_str="", path_results=None):
    """

    :param cells_in_sce:
    :param range_n_clusters:
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

    if testing:
        if shuffling:
            range_n_clusters = [range_n_clusters[0]]
        # else:
        #     range_n_clusters = [n_cluster_var] #np.arange(2, 3)
        # nb of time to apply one given number of cluster
        n_trials = 100
        if shuffling:
            n_shuffle = 100
        else:
            n_shuffle = 1
        silhouettes_shuffling = np.zeros(n_shuffle * range_n_clusters[0])  # *n_trials*range_n_clusters[0])
        for shuffle_index in np.arange(n_shuffle):
            if shuffling:
                m_sces = np.copy(original_m_sces)
                for j in np.arange(len(m_sces[0])):
                    m_sces[:, j] = np.random.permutation(m_sces[:, j])
                for i, s in enumerate(m_sces):
                    # print(f"pos before permutation {np.where(s)[0]}")
                    m_sces[i] = np.random.permutation(s)

            for n_clusters in range_n_clusters:
                print(f"n_clusters {n_clusters}")
                best_kmeans = None
                silhouette_avgs = np.zeros(n_trials)
                best_silhouettes_clusters_avg = None
                max_local_clusters_silhouette = 0
                best_median_silhouettes = 0
                silhouettes_clusters_avg = []
                for trial in np.arange(n_trials):
                    # co_var = np.cov(m_sces)
                    kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
                    # print(f'kmeans.labels_ {kmeans.labels_}')
                    cluster_labels = kmeans.labels_
                    # print(f"len(cluster_labels) {len(cluster_labels)}, "
                    #       f"cluster_labels {cluster_labels}")
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
                    med = np.median(local_clusters_silhouette)
                    if med > best_median_silhouettes:
                        best_median_silhouettes = med
                        best_silhouettes_clusters_avg = local_clusters_silhouette

                    max_local = med  # np.max(local_clusters_silhouette)  # silhouette_avg
                    # TO display, we keep the group with the cluster with the max silhouette
                    if (best_kmeans is None) or (med > max_local_clusters_silhouette):
                        max_local_clusters_silhouette = max_local
                        best_kmeans = kmeans
                        nth_best_list = []
                        count_clusters = nth_best_clusters
                        if count_clusters == -1:
                            count_clusters = n_clusters
                        for b in np.arange(count_clusters):
                            # print(f'local_clusters_silhouette {local_clusters_silhouette}')
                            arg = np.argmax(local_clusters_silhouette)
                            # print(f'local_clusters_silhouette[arg] {local_clusters_silhouette[arg]}')
                            # print(f'[cluster_labels == arg] {[cluster_labels == arg]}, '
                            #       f'cluster_labels {cluster_labels}')
                            # TODO: put neurons list instead of SCEs
                            nth_best_list.append(np.arange(len(m_sces))[cluster_labels == arg])
                            local_clusters_silhouette[arg] = -1
                        dict_best_clusters[n_clusters] = nth_best_list
                    # silhouettes_clusters_avg.extend(sample_silhouette_values)
                    used = False
                    if used:
                        print(f"Silhouettes: {sample_silhouette_values}")
                if shuffling:
                    print(f'end shuffling {shuffle_index}')
                    index = shuffle_index * n_clusters
                    silhouettes_shuffling[index:index + n_clusters] = best_silhouettes_clusters_avg
                else:
                    # if best_kmeans is None:
                    #     continue
                    # print(f'n_clusters {n_clusters}, avg-avg: {np.round(np.mean(silhouette_avgs), 4)}, '
                    #       f'median-avg {np.round(np.median(silhouette_avgs), 4)}, '
                    #       f'median-all {np.round(np.median(silhouettes_clusters_avg), 4)}')
                    # print(f'n_clusters {n_clusters}, silhouettes_clusters_avg {silhouettes_clusters_avg}')
                    cluster_labels_for_neurons[n_clusters] = \
                        find_cluster_labels_for_neurons(cells_in_peak=cells_in_sce,
                                                        cluster_labels=best_kmeans.labels_)
                    if plot_matrix:
                        show_co_var_first_matrix(cells_in_peak=np.copy(cells_in_sce), m_sces=m_sces,
                                                 n_clusters=n_clusters, kmeans=best_kmeans,
                                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_clusters],
                                                 data_str=data_str, path_results=path_results)
        if shuffling:
            # silhouettes_shuffling contains the mean silhouettes values of all clusters produced (100*k)
            p_95 = np.percentile(silhouettes_shuffling, 99)
            print(f'95th p= {p_95}')
    # m_sces = original_m_sces
    return dict_best_clusters, cluster_labels_for_neurons


def show_co_var_first_matrix(cells_in_peak, m_sces, n_clusters, kmeans, cluster_labels_for_neurons,
                             data_str, path_results=None, show_fig=False):
    # cellsinpeak: np.shape(value): (180, 285)
    # fig = plt.figure(figsize=[12, 8])
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex=True,
                                   gridspec_kw={'height_ratios': [1], 'width_ratios': [4, 10]},
                                   figsize=(15, 8))
    # ax1 = plt.subplot(121)
    plt.title(f"{data_str} {n_clusters} clusters")
    # list of size nb_neurons, each neuron having a value from 0 to k clusters
    cluster_labels = kmeans.labels_
    # contains the neurons from the SCE, but ordered by cluster
    # print(f'// np.shape(m_sces) {np.shape(m_sces)}')
    ordered_m_sces = np.zeros((np.shape(m_sces)[0], np.shape(m_sces)[1]))
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_m_sces[start:start + nb_k, :] = m_sces[e, :]
        ordered_m_sces[:, start:start + nb_k] = m_sces[:, e]
        start += nb_k

    co_var = np.corrcoef(ordered_m_sces)  # cov
    # sns.set()
    result = sns.heatmap(co_var, cmap="jet", ax=ax1)  # , vmin=0, vmax=1) YlGnBu
    # ax1.xaxis.get_majorticklabels().set_rotation(90)
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax1.yaxis.get_majorticklabels(), rotation=0)
    ax1.xaxis.set_tick_params(labelsize=5)
    ax1.yaxis.set_tick_params(labelsize=5)
    # ax.invert_yaxis()

    # plt.show()
    original_cells_in_peak = cells_in_peak

    cells_in_peak = np.copy(original_cells_in_peak)
    # ax2 = plt.subplot(1, 2, 2)

    # first order assemblies
    ordered_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]))
    # then order neurons
    ordered_n_cells_in_peak = np.zeros((np.shape(cells_in_peak)[0], np.shape(cells_in_peak)[1]))
    start = 0
    for k in np.arange(n_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        start += nb_k

    start = 0

    for k in np.arange(-1, np.max(cluster_labels_for_neurons) + 1):
        e = np.equal(cluster_labels_for_neurons, k)
        nb_k = np.sum(e)
        # print(f'nb_k {nb_k}, k: {k}')
        ordered_n_cells_in_peak[start:start + nb_k, :] = ordered_cells_in_peak[e, :]
        start += nb_k

    sns.heatmap(ordered_n_cells_in_peak, cbar=False, ax=ax2)
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
    plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)

    # ax.invert_yaxis()
    if path_results is not None:
        fig.savefig(f'{path_results}/{n_clusters}_clusters_{data_str}.pdf',
                    format="pdf")
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
