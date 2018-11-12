import numpy as np
import matplotlib.pyplot as plt


def plot_hist_with_first_perc_and_eb(distribution, threshold_perc, eb_values, bin_factor, mouse_session, title,
                                     x_label, y_label,
                                     filename, mean_eb_values=False, max_range=None):
    # TODO: put it in the minning library and add the two formula to calculate the number of bins from the diap
    # intelligent data analysis Borgelt diap
    distribution = np.array(distribution)
    if isinstance(eb_values, int) or isinstance(eb_values, float):
        eb_values = np.array([eb_values])
    else:
        if mean_eb_values:
            eb_values = np.array([np.mean(eb_values)])
    # trick used to color the last part of the hist
    hist_color = "black"
    perc_color = "red"
    if threshold_perc < 0:
        threshold_perc = 100 + threshold_perc
        hist_color = "red"
        perc_color = "black"
    ms = mouse_session
    # if np.max(distribution) > 4:
    #     max_range = int(np.ceil(np.max(distribution)) + 1)
    # else:
    #     max_range = int(np.ceil(np.max(distribution)) + 0.2)
    max_range = np.max(distribution)
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    fig = plt.figure(figsize=[15, 8])
    ax = plt.subplot(111)
    bins = int(max_range * bin_factor) if max_range > 1 else bin_factor
    hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins,  # range=(0, max_range)
                                                facecolor=hist_color,
                                                weights=weights, log=False)  # density=1) #
    # sns.kdeplot(distribution)
    #  finding the 30% first bins
    sum_perc = 0
    bin_to_colors = []
    last_bin = None
    eb_bins = np.zeros(len(eb_values), dtype="int16")
    width_threshold_bin = 0
    for i, edge in enumerate(edges_plt):
        # print(f"i {i}, edge {edge}")
        if i >= len(hist_plt):
            # means that eb left are on the edge of the last bin
            eb_bins[eb_bins == 0] = i - 1
            break
        if sum_perc < threshold_perc:
            if threshold_perc < (sum_perc + hist_plt[i]):
                last_bin = i
                proportion = (threshold_perc - sum_perc) / hist_plt[i]
                # considering all bin have the same width
                width_threshold_bin = (edges_plt[1] - edges_plt[0]) * proportion
        sum_perc += hist_plt[i]
        if len(eb_values[eb_values <= edge]) > 0:
            # print(f"if edge <= eb_values {edge}")
            if (i + 1) < len(edges_plt):
                eb_bins[eb_values < edges_plt[i + 1]] = i
            else:
                eb_bins[eb_values <= edge] = i
        if sum_perc < threshold_perc:
            bin_to_colors.append(i)
        # if (  is not None) and (sum_perc > threshold_perc):
        #     break

    for i in bin_to_colors:
        patches_plt[i].set_facecolor(perc_color)
    # then coloring part of the last patch:
    if width_threshold_bin > 0:
        rect = patches.Rectangle(xy=(edges_plt[last_bin], 0),
                                 width=width_threshold_bin,
                                 height=hist_plt[last_bin], fill=True, linewidth=0, facecolor=perc_color, zorder=15)
        ax.add_patch(rect)

    # print(f"eb_values {eb_values}, eb_bins {eb_bins}, hist_plt[eb_bins] {hist_plt[eb_bins]}")
    plt.scatter(x=eb_values, y=hist_plt[eb_bins] * 1.1, marker="*", color="green", s=60, zorder=20)

    plt.xlim(0, max_range)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig.savefig(f'{ms.param.path_results}/{filename}_{ms.param.time_str}.png',
                format="png")
    if ms.param.with_svg_format:
        fig.savefig(f'{ms.param.path_results}/{filename}_{ms.param.time_str}.svg',
                    format="svg")
    plt.close()


def plot_hist_clusters_by_sce(cluster_particpation_to_sce, data_str="", save_formats="pdf",
                              param=None, save_plot=True, show_fig=False):
    # key is a binary tuple representing the activity of a cluster in an SCE, and the value is an int representing
    # the number of time this pattern of activity is present in SCE
    tuples_dict = {}
    n_sces = len(cluster_particpation_to_sce[0, :])
    for i in np.arange(n_sces):
        sce_tuple = tuple(cluster_particpation_to_sce[:, i])
        tuples_dict[sce_tuple] = tuples_dict.get(sce_tuple, 0) + 1

    network_events_percentages = np.array(list(tuples_dict.values()))
    network_events_percentages = (network_events_percentages / n_sces) * 100
    distribution = network_events_percentages
    print(f"network_events_percentages {network_events_percentages}")

    max_range = np.max(distribution)
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    fig = plt.figure(figsize=[15, 8])
    ax = plt.subplot(111)
    range_by_bin = 10
    bins = int(100 / range_by_bin)
    hist_plt, edges_plt, patches_plt = plt.hist(distribution, bins=bins, range=(0, 100),
                                                facecolor="blue",
                                                weights=weights, log=False)

    plt.xlim(0, 100)
    xticks_pos = np.arange(0, 100, range_by_bin)
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticks_pos)
    # plt.title(title)
    plt.xlabel("Percentage of network events")
    plt.ylabel("Probability")

    if save_plot and (param is not None):
        # transforming a string in a list
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{data_str}_{param.time_str}.{save_format}', format=f"{save_format}")
    # Display the spike raster plot
    if show_fig:
        plt.show()
    plt.close()


def time_correlation_graph(spike_nums, events_peak_times, data_id, param,
                           title_option="",
                           cells_groups=None, groups_colors=None,
                           time_window=5, set_y_limit_to_max=True,
                           set_x_limit_to_max=True,
                           time_stamps_by_ms=0.01, ms_scale=200, save_formats="pdf"):
    print("time_correlation_graph")
    nb_neurons = len(spike_nums)
    n_times = len(spike_nums[0, :])
    # values for each cell
    time_lags_dict = dict()
    correlation_dict = dict()
    # for ploting
    time_lags_list = []
    correlation_list = []

    for neuron in np.arange(nb_neurons):
        # look at onsets
        neuron_spikes, = np.where(spike_nums[neuron, :])

        if len(neuron_spikes) == 0:
            continue

        spike_nums_to_use = spike_nums

        # time_window by 4
        distribution_array_2_d = np.zeros((nb_neurons, ((time_window * 4) + 1)),
                                          dtype="int16")

        # event_index = time_window
        # looping on each spike of the main neuron
        for n, peak_time in enumerate(events_peak_times):
            # only taking in consideration events that are not too close from bottom range or upper range
            min_limit = max(peak_time - time_window, 0)
            max_limit = min((peak_time + time_window), (n_times - 1))
            if np.sum(spike_nums[neuron, min_limit:max_limit]) == 0:
                continue
            # see to consider the case in which the cell spikes 2 times around a peak during the tim_window
            neuron_spike_time = np.where(spike_nums[neuron, min_limit:max_limit])[0][0]
            spikes_indices = np.where(spike_nums_to_use[:, min_limit:(max_limit + 1)])
            conn_cells_indices = spikes_indices[0]
            spikes_indices = neuron_spike_time - spikes_indices[1]
            spikes_indices += time_window*2
            # print(f"spikes_indices {spikes_indices}")
            distribution_array_2_d[conn_cells_indices, spikes_indices] += 1

        mask = np.ones(nb_neurons, dtype="bool")
        mask[neuron] = False
        # sum of spikes at each times lag
        distribution_array = np.sum(distribution_array_2_d[mask, :], axis=0)
        # print(f"distribution_array {distribution_array}")
        total_spikes = np.sum(distribution_array)
        # adding the cell only if it has at least a spike around peak times
        if total_spikes > 0:
            correlation_value = np.max(distribution_array) / total_spikes
            # array_to_average = np.zeros(np.sum(distribution_array))
            # start = 0
            # for index, time_lag in enumerate(np.arange(-time_window * 2, time_window * 2 + 1)):
            #     n_spike_for_this_time_lag = distribution_array[index]
            #     array_to_average[start:(start+n_spike_for_this_time_lag)] = time_lag
            #     start += n_spike_for_this_time_lag
            # avg_time_lag = np.mean(array_to_average)
            # other way:
            time_lags_range = np.arange(-time_window * 2, time_window * 2 + 1)
            distribution_array = distribution_array * time_lags_range
            avg_time_lag = np.sum(distribution_array)/total_spikes
            time_lags_dict[neuron] = avg_time_lag
            correlation_dict[neuron] = correlation_value

    for cell, time_lag in time_lags_dict.items():
        time_lags_list.append(time_lag)
        correlation_list.append(correlation_dict[cell])

    # print(f"time_lags_list {time_lags_list}")
    # print(f"correlation_list {correlation_list}")

    default_cell_color = "grey"

    fig, ax = plt.subplots(nrows=1, ncols=1,
                           gridspec_kw={'height_ratios': [1]},
                           figsize=(20, 20))

    ax.scatter(time_lags_list, correlation_list, color=default_cell_color, marker="o",
               s=100, zorder=1, alpha=0.5)

    if cells_groups is not None:
        for group_id, cells in enumerate(cells_groups):
            for cell in cells:
                if cell in time_lags_dict:
                    ax.scatter(time_lags_dict[cell], correlation_dict[cell], color=groups_colors[group_id],
                               marker="o",
                               s=240, zorder=1)
                else:
                    print(f"{data_id}: cell {cell} not in time-correlation graph")

    xticks_pos = []
    # display a tick every 200 ms, time being in seconds
    times_for_s_scale = ms_scale * 0.001
    time_window_s = (time_window / time_stamps_by_ms) * 0.001
    xticklabels = []
    labels_in_s = np.arange(-time_window_s * 2, time_window_s * 2 + 1, times_for_s_scale)
    pos_range = np.arange(-time_window * 2, time_window * 2 + 1, ms_scale*time_stamps_by_ms)
    # print(f"max_value {max_value}")
    if set_x_limit_to_max:
        min_range_index = 0
        max_range_index = len(pos_range) - 1
    else:
        max_value = np.max((np.abs(np.min(time_lags_list)), np.abs(np.max(time_lags_list))))
        min_range_index = np.searchsorted(pos_range, -max_value, side='left') - 1
        max_range_index = np.searchsorted(pos_range, max_value, side='right')
    # print(f"min_range_index {min_range_index}, max_range_index {max_range_index}, pos_range {pos_range}")
    labels_in_s = labels_in_s[min_range_index:max_range_index+1]
    for index_pos, pos in enumerate(pos_range[min_range_index:max_range_index+1]):
        xticks_pos.append(pos)
        xticklabels.append(np.round(labels_in_s[index_pos], 1))
    # print(f"xticks_pos {xticks_pos}")
    # print(f"xticklabels {xticklabels}")
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticklabels)
    if set_x_limit_to_max:
        ax.set_xlim(-time_window * 2, (time_window * 2) +1)
    else:
        ax.set_xlim(pos_range[min_range_index], pos_range[max_range_index])
    if set_y_limit_to_max:
        ax.set_ylim(0, 1)

    ax.set_xlabel("Time lag (s)")
    ax.set_ylabel("Correlation")

    plt.title(f"Time-correlation graph {data_id} {title_option}")

    #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
    # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,

    if isinstance(save_formats, str):
        save_formats = [save_formats]
    for save_format in save_formats:
        fig.savefig(f'{param.path_results}/{data_id}_time-correlation-graph_{title_option}'
                    f'_{param.time_str}.{save_format}',
                    format=f"{save_format}")
    plt.close()
