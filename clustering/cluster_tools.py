import numpy as np


def detect_cluster_activations_with_sliding_window(spike_nums, window_duration, cluster_labels,
                                                   sce_times_numbers,
                                                   perc_threshold=50,
                                                   non_binary=False, debug_mode=False):
    # return a binary 2D-array, with lines representing cells, and columns time, 1 means the cluster is active
    # at that time (meaning > 50% of cells spike at least one time during the sliding window). Meaning 1 should be put
    # for all cells of cluster each time, and should last at least the time of a sliding window
    # cluster_labels is an array of length len(spike_nums), each index represent a cell in the same corresponding
    # order than spike_nums, and the value represent the clusters number.
    # sce_times_numbers: an array of len n_times, that for each times give the SCE number or -1 if part of no SCE,
    # return by the function detect_sce_with_sliding_window


    if non_binary:
        binary_spikes = np.zeros((len(spike_nums), len(spike_nums[0, :])), dtype="int8")
        for neuron, spikes in enumerate(spike_nums):
            binary_spikes[neuron, spikes > 0] = 1
        spike_nums = binary_spikes

    # first counting how many cells are part of a cluster
    n_cells_in_a_cluster = 0
    n_clusters=0
    # print(f"np.max(cluster_labels) {np.max(cluster_labels)} cluster_labels {cluster_labels}")
    for cluster_number in np.arange(np.max(cluster_labels) + 1):
        # boolean array, True are the cells part of the cluster cluster_number
        cells_in_cluster_bool = np.equal(cluster_labels, cluster_number)
        nb_cells_in_cluster = np.sum(cells_in_cluster_bool)
        if nb_cells_in_cluster > 0:
            n_cells_in_a_cluster += nb_cells_in_cluster
            n_clusters += 1

    if n_clusters == 0:
        print('detect_cluster_activations_with_sliding_window no cluster')
        return

    n_cells = len(spike_nums)
    n_times = len(spike_nums[0, :])
    n_sces = np.max(sce_times_numbers) + 1

    if debug_mode:
        print(f"n_cells_in_a_cluster {n_cells_in_a_cluster} n_clusters {n_clusters} n_sces {n_sces}")

    clusters_activations = np.zeros((n_cells_in_a_cluster, n_times), dtype="uint8")
    clusters_activations_by_cluster = np.zeros((n_clusters, n_times), dtype="uint8")
    cluster_particpation_to_sce = np.zeros((n_clusters, n_sces), dtype="uint8")
    # n_clusters could be only 2, but the cells could belong to cluster numbers that goes more than 2, even there will
    # be only 2 clusters at the end
    max_cluster_number = np.max(cluster_labels)
    clusters_corresponding_index = np.zeros(max_cluster_number+1,  dtype="uint8")
    # if the value stays at -1, it means the corresponding clusters has not cells associated and so no new cluster
    clusters_corresponding_index = clusters_corresponding_index - 1

    start_time_active_cluster = -1
    cluster_start_index = 0
    # keep track to how many cluster have been checked so far, the number correspond to the index of the cluster
    # used for cluster_particpation_to_sce
    significant_cluster_number = -1
    # looping to each cluster
    # we skip -1 value, as it means the cell is part of no cluster
    for cluster_number in np.arange(np.max(cluster_labels) + 1):
        # boolean array, True are the cells part of the cluster cluster_number
        cells_in_cluster_bool = np.equal(cluster_labels, cluster_number)
        nb_cells_in_cluster = np.sum(cells_in_cluster_bool)
        if nb_cells_in_cluster == 0:
            continue

        significant_cluster_number += 1
        if debug_mode:
            print(f"cluster_number {cluster_number}, significant_cluster_number {significant_cluster_number}, "
                  f"nb_cells_in_cluster {nb_cells_in_cluster}")
        # looping through the window
        for t in np.arange(0, (n_times - window_duration)):
            # if debug_mode:
            #     if t % 10 ** 6 == 0:
            #         print(f"t {t}")
            sum_spikes = np.sum(spike_nums[cells_in_cluster_bool, t:(t + window_duration)], axis=1)
            # neurons with sum > 1 are active during a SCE
            nb_cells_active_in_window = len(np.where(sum_spikes)[0])

            # print(f"nb_cells_active_in_window {nb_cells_active_in_window}")
            # if the more than 50% of the cell from this cluster is active, then
            # we decide that this cluster is active during this time_window
            if nb_cells_active_in_window >= (nb_cells_in_cluster*0.5):
                if start_time_active_cluster == -1:
                    start_time_active_cluster = t
            else:
                if start_time_active_cluster > -1:
                    # then a new active cluster period is detected
                    beg = cluster_start_index
                    end_cell = cluster_start_index+nb_cells_in_cluster
                    if debug_mode:
                        print(f"new cluster activation: start_time_active_cluster {start_time_active_cluster}  t {t} ")
                    clusters_activations[beg:end_cell, start_time_active_cluster:t] = 1
                    clusters_activations_by_cluster[significant_cluster_number, start_time_active_cluster:t] = 1
                    # then checking if any of this time is part of an SCE
                    times_part_of_sce = np.where(sce_times_numbers[start_time_active_cluster:t] > -1)[0]
                    if len(times_part_of_sce) > 0:
                        # getting the different SCE index, if more than one is inside
                        sce_indices = np.unique(sce_times_numbers[times_part_of_sce+start_time_active_cluster])
                        if debug_mode:
                            print(f"sce_indices {sce_indices}")
                        cluster_particpation_to_sce[significant_cluster_number, sce_indices] = 1
                    start_time_active_cluster = -1
        clusters_corresponding_index[cluster_number] = significant_cluster_number
        cluster_start_index += nb_cells_in_cluster

    return clusters_activations, clusters_activations_by_cluster, cluster_particpation_to_sce, clusters_corresponding_index
