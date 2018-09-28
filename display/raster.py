import matplotlib.pyplot as plt
import numpy as np
import math


def plot_spikes_raster(spike_nums, param=None, title=None, file_name=None,
                       spike_train_format=False,
                       y_ticks_labels=None,
                       y_ticks_labels_size=None,
                       save_raster=False,
                       show_raster=False,
                       plot_with_amplitude=False,
                       activity_threshold=None,
                       save_formats="png",
                       span_area_coords=None,
                       span_area_colors=None,
                       cells_to_highlight=None,
                       cells_to_highlight_colors=None,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=False,
                       span_cells_to_highlight=True,
                       spike_shape="|",
                       spike_shape_size=10,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       seq_times_to_color_dict=None,
                       seq_colors=None
                       ):
    """
    
    :param spike_nums: np.array of 2D, axis=1 (lines) represents the cells, the columns representing the spikes
    It could be binary, or containing the amplitude, if amplitudes values should be display put plot_with_amplitude
    to True
    :param param:
    :param spike_train_format: if True, means the data is a list of np.array, and then spike_nums[i][j] is
    a timestamps value as float
    :param title: 
    :param file_name: 
    :param save_raster: 
    :param show_raster: 
    :param plot_with_amplitude: 
    :param activity_threshold: 
    :param save_formats: 
    :param span_area_coords: List of list of tuples of two float representing coord of are to span with a color
    corresponding to the one in span_area_colors
    :param span_area_colors: list of colors, same len as span_area_coords
    :param cells_to_highlight: cells index to span and with special spikes color, list of int
    :param cells_to_highlight_colors: cells colors to span, same len as cells_to_span, list of string
    :param raster_face_color:
    :param cell_spikes_color:
    :param spike_shape: shape of the spike, "|", "*", "o"
    :param spike_shape_size: use for shape != of "|"
    :param seq_times_to_color_dict: None or a dict with as the key a tuple of int representing the cell index,
    and as a value a list of set, each set composed of int representing the times value at which the cell spike should
    be colored. It will be colored if there is indeed a spike at that time otherwise, the default color will be used.
    :param seq_colors: A dict, with key a tuple represening the indices of the seq and as value of colors,
    a color, should have the same keys as seq_times_to_color_dict
    :return: 
    """

    if spike_nums is None:
        return

    if plot_with_amplitude and spike_train_format:
        # not possible so far
        return

    n_cells = len(spike_nums)

    if not plot_with_amplitude:
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True,
                                       gridspec_kw={'height_ratios': [10, 2]},
                                       figsize=(15, 8))
        fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
    else:
        fig = plt.figure(figsize=(15, 8))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 1})
        outer = gridspec.GridSpec(1, 2, width_ratios=[100, 1])  # , wspace=0.2, hspace=0.2)

    if plot_with_amplitude:
        inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                 subplot_spec=outer[0], height_ratios=[10, 2])
        # inner.tight_layout(fig, pad=0.1)
        ax1 = fig.add_subplot(inner[0])  # plt.Subplot(fig, inner[0])
        min_value = np.min(spike_nums)
        max_value = np.max(spike_nums)
        step_color_value = 0.1
        colors = np.r_[np.arange(min_value, max_value, step_color_value)]
        mymap = plt.get_cmap("jet")
        # get the colors from the color map
        my_colors = mymap(colors)

        # colors = plt.cm.hsv(y / float(max(y)))
        scalar_map = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=min_value, vmax=max_value))
        # # fake up the array of the scalar mappable. Urgh…
        scalar_map._A = []

    # -------- end plot with amplitude ---------

    ax1.set_facecolor(raster_face_color)

    min_time = 0
    max_time = 0

    for y, neuron in enumerate(spike_nums):
        if spike_train_format:
            if y == 0:
                min_time = np.min(neuron)
            else:
                min_time = int(np.min((min_time, np.min(neuron))))
            max_time = int(np.ceil(np.max((max_time, np.max(neuron)))))
        # print(f"Neuron {y}, total spikes {len(np.where(neuron)[0])}, "
        #       f"nb > 2: {len(np.where(neuron>2)[0])}, nb < 2: {len(np.where(neuron[neuron<2])[0])}")
        color_neuron = cell_spikes_color
        if cells_to_highlight is not None:
            if y in cells_to_highlight:
                cells_to_highlight = np.array(cells_to_highlight)
                index = np.where(cells_to_highlight == y)[0][0]
                color_neuron = cells_to_highlight_colors[index]
        if spike_train_format:
            neuron_times = neuron
        else:
            neuron_times = np.where(neuron)[0]
        if spike_shape != "|":
            if plot_with_amplitude:
                ax1.scatter(neuron_times, np.repeat(y, len(neuron_times)), color=scalar_map.to_rgba(neuron[neuron > 0]),
                            marker=spike_shape,
                            s=spike_shape_size)
            else:
                ax1.scatter(neuron_times, np.repeat(y, len(neuron_times)), color=color_neuron, marker=spike_shape,
                            s=spike_shape_size)
        else:
            if plot_with_amplitude:
                ax1.vlines(neuron_times, y - .5, y + .5, color=scalar_map.to_rgba(neuron[neuron > 0]),
                           linewidth=1)
            else:
                ax1.vlines(neuron_times, y - .5, y + .5, color=color_neuron, linewidth=1)

    if seq_times_to_color_dict is not None:
        for seq_indices, seq_times_list in seq_times_to_color_dict.items():
            for times_list_index, times_list in enumerate(seq_times_list):
                for time_index, t in enumerate(times_list):
                    cell_index = seq_indices[time_index]
                    # first we make sure the cell does spike at the given time
                    if spike_train_format:
                        if t not in spike_nums[cell_index]:
                            continue
                    else:
                        if spike_nums[cell_index, t] == 0:
                            # print(f"Not there: seq {times_list_index} cell {cell_index}, time {t}")
                            continue
                        # print(f"## There: seq {times_list_index} cell {cell_index}, time {t}")
                    # if so, we draw the spike
                    if spike_shape != "|":
                        ax1.scatter(t, cell_index, color=seq_colors[seq_indices],
                                    marker=spike_shape,
                                    s=spike_shape_size)
                    else:
                        ax1.vlines(t, cell_index - .5, cell_index + .5, color=seq_colors[seq_indices],
                                   linewidth=1)
    if spike_train_format:
        n_times = int(math.ceil(max_time - min_time))
    else:
        n_times = len(spike_nums[0, :])

    # draw span to highlight some periods
    if span_area_coords is not None:
        if len(span_area_coords) != len(span_area_colors):
            raise Exception("span_area_coords and span_area_colors are not the same size")
        for index, span_area_coord in enumerate(span_area_coords):
            for coord in span_area_coord:
                if span_area_colors is not None:
                    color = span_area_colors[index]
                else:
                    color = "lightgrey"
                ax1.axvspan(coord[0], coord[1], alpha=0.5, facecolor=color)

    if (cells_to_highlight is not None) and span_cells_to_highlight:
        for index, cell_to_span in enumerate(cells_to_highlight):
            ax1.axhspan(cell_to_span - 0.5, cell_to_span + 0.5, alpha=0.4, facecolor=cells_to_highlight_colors[index])

    ax1.set_ylim(-1, len(spike_nums))
    if y_ticks_labels is not None:
        ax1.set_yticks(np.arange(len(spike_nums)))
        ax1.set_yticklabels(y_ticks_labels)
    if y_ticks_labels_size is not None:
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)

    if spike_train_format:
        ax1.set_xlim(min_time - 1, max_time + 1)
    else:
        ax1.set_xlim(-1, len(spike_nums[0, :]) + 1)
    # ax1.margins(x=0, tight=True)

    if title is None:
        ax1.set_title('Spikes raster plot')
    else:
        ax1.set_title(title)
    # Give x axis label for the spike raster plot
    # ax.xlabel('Frames')
    # Give y axis label for the spike raster plot
    ax1.set_ylabel('Cells (#)')

    if sliding_window_duration >= 1:
        print("sliding_window_duration > 1")
        sum_spikes = np.zeros(n_times)
        if spike_train_format:
            windows_sum = np.zeros((n_cells, n_times), dtype="int16")
            # one cell can participate to max one spike by window
            # if value is True, it means this cell has already been counted
            cell_window_participation = np.zeros((n_cells, n_times), dtype="bool")
            for cell, spikes_train in enumerate(spike_nums):
                for spike_time in spikes_train:
                    # first determining to which windows to add the spike
                    spike_index = int(spike_time - min_time)
                    first_index_window = np.max((0, spike_index - sliding_window_duration))
                    if np.sum(cell_window_participation[cell, first_index_window:spike_index]) == 0:
                        windows_sum[cell, first_index_window:spike_index] += 1
                        cell_window_participation[cell, first_index_window:spike_index] = True
                    else:
                        for t in np.arange(first_index_window, spike_index):
                            if cell_window_participation[cell, t] is False:
                                windows_sum[cell, t] += 1
                                cell_window_participation[cell, t] = True
            sum_spikes = np.sum(windows_sum, axis=0)
            print("sliding window over")
            # for index, t in enumerate(np.arange(int(min_time), int((np.ceil(max_time) - sliding_window_duration)))):
            #     # counting how many cell fire during that window
            #     if (index % 1000) == 0:
            #         print(f"index {index}")
            #     sum_value = 0
            #     t_min = t
            #     t_max = t + sliding_window_duration
            #     for spikes_train in spike_nums:
            #         # give the indexes
            #         # np.where(np.logical_and(spikes_train >= t, spikes_train < t_max))
            #         spikes = spikes_train[np.logical_and(spikes_train >= t, spikes_train < t_max)]
            #         nb_spikes = len(spikes)
            #         if nb_spikes > 0:
            #             sum_value += 1
            #     sum_spikes[index] = sum_value
            # sum_spikes[(n_times - sliding_window_duration):] = sum_value
        else:
            for t in np.arange(0, (n_times - sliding_window_duration)):
                # One spike by cell max in the sum process
                sum_value = np.sum(spike_nums[:, t:(t + sliding_window_duration)], axis=1)
                sum_spikes[t] = len(np.where(sum_value)[0])
            sum_spikes[(n_times - sliding_window_duration):] = len(np.where(sum_value)[0])
    else:
        if spike_train_format:
            pass
        else:
            binary_spikes = np.zeros((n_cells, n_times), dtype="int8")
            for neuron, spikes in enumerate(spike_nums):
                binary_spikes[neuron, spikes > 0] = 1
            if param.bin_size > 1:
                sum_spikes = np.mean(np.split(np.sum(binary_spikes, axis=0), n_times // param.bin_size), axis=1)
                sum_spikes = np.repeat(sum_spikes, param.bin_size)
            else:
                sum_spikes = np.sum(binary_spikes, axis=0)

    if spike_train_format:
        x_value = np.arange(min_time, max_time)
    else:
        x_value = np.arange(n_times)

    if plot_with_amplitude:
        ax2 = fig.add_subplot(inner[1], sharex=ax1)

    # sp = UnivariateSpline(x_value, sum_spikes, s=240)
    # ax2.fill_between(x_value, 0, smooth_curve(sum_spikes), facecolor="black") # smooth_curve(sum_spikes)
    if show_sum_spikes_as_percentage:
        print("using percentages")
        sum_spikes = sum_spikes / n_cells
        sum_spikes *= 100
        if activity_threshold is not None:
            activity_threshold = activity_threshold / n_cells
            activity_threshold *= 100

    ax2.fill_between(x_value, 0, sum_spikes, facecolor="black")
    if activity_threshold is not None:
        line_beg_x = 0
        line_end_x = 0
        if spike_train_format:
            line_beg_x = min_time - 1
            line_end_x = max_time + 1
        else:
            line_beg_x = -1
            line_end_x = len(spike_nums[0, :]) + 1
        ax2.hlines(activity_threshold, line_beg_x, line_end_x, color="red", linewidth=2, linestyles="dashed")

    # draw span to highlight some periods
    if span_area_coords is not None:
        for index, span_area_coord in enumerate(span_area_coords):
            for coord in span_area_coord:
                if span_area_colors is not None:
                    color = span_area_colors[index]
                else:
                    color = "lightgrey"
                ax2.axvspan(coord[0], coord[1], alpha=0.5, facecolor=color)

    # early born
    if cells_to_highlight is not None:
        for index, cell_to_span in enumerate(cells_to_highlight):
            ax2.vlines(np.where(spike_nums[cell_to_span, :])[0], 0, np.max(sum_spikes),
                       color=cells_to_highlight_colors[index],
                       linewidth=2, linestyles="dashed", alpha=0.2)

    # ax2.yaxis.set_visible(False)
    ax2.set_frame_on(False)
    ax2.get_xaxis().set_visible(True)
    if spike_train_format:
        ax2.set_xlim(min_time - 1, max_time + 1)
    else:
        ax2.set_xlim(-1, len(spike_nums[0, :]) + 1)

    # print(f"max sum_spikes {np.max(sum_spikes)}, mean  {np.mean(sum_spikes)}, median {np.median(sum_spikes)}")
    ax2.set_ylim(0, np.max(sum_spikes))

    # color bar section
    if plot_with_amplitude:
        inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                   subplot_spec=outer[1])  # , wspace=0.1, hspace=0.1)
        ax3 = fig.add_subplot(inner_2[0])  # plt.Subplot(fig, inner_2[0])
        fig.colorbar(scalar_map, cax=ax3)

    if save_raster and (param is not None):
        # transforming a string in a list
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{file_name}_{param.time_str}.{save_format}', format=f"{save_format}")
    # Display the spike raster plot
    if show_raster:
        plt.show()
    plt.close()
