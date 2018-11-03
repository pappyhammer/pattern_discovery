import matplotlib.pyplot as plt
import numpy as np
import math


def plot_dendogram_from_fca(cluster_tree, nb_cells, save_plot, axes_list=None, fig_to_use=None, file_name="",
                            dendo_face_color='black',
                            default_line_color='white',
                            cell_labels=None,
                            param=None,
                            show_plot=False, save_formats="pdf"):
    """

    :param current_cluster:
    :param merge_history: list of list, each list is composed of 3 elemnts, the two first are the ones merged (could
    be a list or an int) and the last value is a float representing the scale value
    :return:
    """
    # scale value under 1 are not significant
    if fig_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                figsize=(15, 8))

        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 1})
    else:
        fig = fig_to_use
        ax1 = axes_list[0]

    ax1.set_facecolor(dendo_face_color)

    max_y = cluster_tree.max_y_pos

    cluster_tree.plot_cluster(ax=ax1, with_scale_value=True, default_line_color=default_line_color)

    ax1.hlines(cluster_tree.significant_threshold, 0, nb_cells - 1, color=default_line_color, linewidth=2,
               linestyles="dashed")

    ax1.set_xticks(np.arange(nb_cells))
    if cell_labels is None:
        ax1.set_xticklabels(cluster_tree.pos_cells)
    else:
        ticks_labels = []
        for pos in cluster_tree.pos_cells:
            ticks_labels.append(cell_labels[pos])

        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        # plt.setp(ax1, facecolor='black')
        # ax1.xaxis.get_majorticklabels().set_rotation(45)
        ax1.xaxis.set_tick_params(labelsize=8)
        ax1.set_xticklabels(ticks_labels)
    ax1.xaxis.set_ticks_position('none')
    ax1.yaxis.set_ticks_position('none')
    ax1.set_yticklabels([])
    ax1.set_xlim(-1, nb_cells + 1)
    ax1.set_ylim(0, max_y + 1)
    # ax1.set_frame_on(False)

    if save_plot and (param is not None):
        # transforming a string in a list
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{file_name}_{param.time_str}.{save_format}', format=f"{save_format}")
    # Display the spike raster plot
    if show_plot:
        plt.show()
    plt.close()


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
                       span_area_only_on_raster=True,
                       cells_to_highlight=None,
                       cells_to_highlight_colors=None,
                       color_peaks_activity=False,
                       horizontal_lines=None,
                       horizontal_lines_colors=None,
                       horizontal_lines_sytle=None,
                       horizontal_lines_linewidth=None,
                       vertical_lines=None,
                       vertical_lines_colors=None,
                       vertical_lines_sytle=None,
                       vertical_lines_linewidth=None,
                       sliding_window_duration=1,
                       show_sum_spikes_as_percentage=False,
                       span_cells_to_highlight=None,
                       span_cells_to_highlight_colors=None,
                       spike_shape="|",
                       spike_shape_size=10,
                       raster_face_color='black',
                       cell_spikes_color='white',
                       seq_times_to_color_dict=None,
                       link_seq_categories=None,
                       link_seq_color=None, min_len_links_seq=3,
                       link_seq_line_width=1, link_seq_alpha=1,
                       jitter_links_range=1,
                       display_link_features=True,
                       seq_colors=None, debug_mode=False,
                       axes_list=None,
                       SCE_times=None,
                       ylabel="Cells (#)",
                       without_activity_sum=False
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
    :param color_peaks_activity: if True, will span to the color of cells_to_highlight_colors each time at which a cell
    among cells_to_highlight will spike on the actiivty peak diagram
    :param horizontal_lines: list of float, representing the y coord at which trace horizontal lines
    :param horizontal_lines_colors: if horizontal_lines is not None, will set the colors of each line,
    list of string or color code
    :param horizontal_lines_style: give the style of the lines, string
    :param vertical_lines: list of float, representing the x coord at which trace vertical lines
    :param vertical__lines_colors: if horizontal_lines is not None, will set the colors of each line,
    list of string or color code
    :param vertical__lines_style: give the style of the lines, string
    :param vertical_lines_linewidth:
    :param raster_face_color:
    :param cell_spikes_color:
    :param spike_shape: shape of the spike, "|", "*", "o"
    :param spike_shape_size: use for shape != of "|"
    :param seq_times_to_color_dict: None or a dict with as the key a tuple of int representing the cell index,
    and as a value a list of set, each set composed of int representing the times value at which the cell spike should
    be colored. It will be colored if there is indeed a spike at that time otherwise, the default color will be used.
    :param seq_colors: A dict, with key a tuple represening the indices of the seq and as value of colors,
    a color, should have the same keys as seq_times_to_color_dict
    :param link_seq_color: if not None, give the color with which link the spikes from a sequence. If not None,
    seq_colors will be ignored
    :param min_len_links_seq: minimum len of a seq for the links to be drawn
    :param axes_list if not None, give a list of axes that will be used, and be filled, but no figure will be created
    or saved then. Doesn't work yet is show_amplitude is True
    :param SCE_times:  a list of tuple corresponding to the first and last index of each SCE,
    (last index being included in the SCE). Will display the position of the SCE and their number above the activity
    diagram. If None, the overall time will be displayed. Need to be adapted to the format spike_numw or
    spike_train
    :param without_activity_sum: if True, don't plot the sum of activity diagram, valid only if axes_list is not None
    :return: 
    """

    if spike_nums is None:
        return

    if plot_with_amplitude and spike_train_format:
        # not possible so far
        return

    n_cells = len(spike_nums)
    if axes_list is None:
        if not plot_with_amplitude:
            sharex = False if (SCE_times is None) else True
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=sharex,
                                           gridspec_kw={'height_ratios': [10, 2]},
                                           figsize=(15, 8))
            fig.set_tight_layout({'rect': [0, 0, 1, 0.95], 'pad': 1.5, 'h_pad': 1.5})
        else:
            fig = plt.figure(figsize=(15, 8))
            fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 1})
            outer = gridspec.GridSpec(1, 2, width_ratios=[100, 1])  # , wspace=0.2, hspace=0.2)
    else:
        if without_activity_sum:
            ax1 = axes_list[0]
        else:
            ax1, ax2 = axes_list

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
                            s=spike_shape_size, zorder=20)
            else:
                ax1.scatter(neuron_times, np.repeat(y, len(neuron_times)), color=color_neuron, marker=spike_shape,
                            s=spike_shape_size, zorder=20)
        else:
            if plot_with_amplitude:
                ax1.vlines(neuron_times, y - .5, y + .5, color=scalar_map.to_rgba(neuron[neuron > 0]),
                           linewidth=1, zorder=20)
            else:
                ax1.vlines(neuron_times, y - .5, y + .5, color=color_neuron, linewidth=1, zorder=20)

    if seq_times_to_color_dict is not None:
        seq_count = 0
        links_labels = []
        links_labels_color = []
        links_labels_y_coord = []
        nb_jitters = 10
        indices_rand_x = np.linspace(-jitter_links_range, jitter_links_range, nb_jitters)
        np.random.shuffle(indices_rand_x)
        for seq_indices, seq_times_list in seq_times_to_color_dict.items():
            nb_seq_times = 0
            for times_list_index, times_list in enumerate(seq_times_list):
                x_coord_to_link = []
                y_coord_to_link = []
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
                    if link_seq_color is not None:
                        x_coord_to_link.append(t)
                        y_coord_to_link.append(cell_index)
                    else:
                        # if so, we draw the spike
                        if spike_shape != "|":
                            ax1.scatter(t, cell_index, color=seq_colors[seq_indices],
                                        marker=spike_shape,
                                        s=spike_shape_size, zorder=20)
                        else:
                            ax1.vlines(t, cell_index - .5, cell_index + .5, color=seq_colors[seq_indices],
                                       linewidth=1, zorder=20)
                if (link_seq_color is not None) and (len(x_coord_to_link) >= min_len_links_seq):
                    if isinstance(link_seq_color, str):
                        color_to_use = link_seq_color
                    else:
                        color_to_use = link_seq_color[seq_count % len(link_seq_color)]
                    x_coord_to_link = np.array(x_coord_to_link)
                    ax1.plot(x_coord_to_link + indices_rand_x[seq_count % nb_jitters], y_coord_to_link,
                             color=color_to_use,
                             linewidth=link_seq_line_width, zorder=30, alpha=link_seq_alpha)
                    nb_seq_times += 1
            if nb_seq_times > 0:
                category = ""
                if link_seq_categories is not None:
                    category = "*" * link_seq_categories[seq_indices]
                links_labels.append(f"l{len(seq_indices)}, r{nb_seq_times} {category}")
                links_labels_color.append(color_to_use)
                links_labels_y_coord.append((seq_indices[0] + seq_indices[-1]) / 2)
            seq_count += 1

    ax1.set_ylim(-1, len(spike_nums))
    if y_ticks_labels is not None:
        ax1.set_yticks(np.arange(len(spike_nums)))
        ax1.set_yticklabels(y_ticks_labels)
    if y_ticks_labels_size is not None:
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
    else:
        if len(spike_nums) < 50:
            y_ticks_labels_size = 5
        elif len(spike_nums) < 100:
            y_ticks_labels_size = 4
        elif len(spike_nums) < 200:
            y_ticks_labels_size = 3
        elif len(spike_nums) < 400:
            y_ticks_labels_size = 2
        else:
            y_ticks_labels_size = 1
        ax1.yaxis.set_tick_params(labelsize=y_ticks_labels_size)

    if seq_times_to_color_dict is not None:
        if link_seq_color is not None:
            ax_right = ax1.twinx()
            ax_right.set_frame_on(False)
            ax_right.set_ylim(-1, len(spike_nums))
            ax_right.set_yticks(links_labels_y_coord)
            # clusters labels
            ax_right.set_yticklabels(links_labels)
            ax_right.yaxis.set_ticks_position('none')
            if y_ticks_labels_size > 1:
                y_ticks_labels_size -= 1
            else:
                y_ticks_labels_size -= 0.5
            ax_right.yaxis.set_tick_params(labelsize=y_ticks_labels_size)
            # ax_right.yaxis.set_tick_params(labelsize=2)
            for index in np.arange(len(links_labels)):
                ax_right.get_yticklabels()[index].set_color(links_labels_color[index])

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
                ax1.axvspan(coord[0], coord[1], alpha=0.5, facecolor=color, zorder=1)

    if (span_cells_to_highlight is not None):
        for index, cell_to_span in enumerate(span_cells_to_highlight):
            ax1.axhspan(cell_to_span - 0.5, cell_to_span + 0.5, alpha=0.4,
                        facecolor=span_cells_to_highlight_colors[index])

    if horizontal_lines is not None:
        line_beg_x = 0
        line_end_x = 0
        if spike_train_format:
            line_beg_x = min_time - 1
            line_end_x = max_time + 1
        else:
            line_beg_x = -1
            line_end_x = len(spike_nums[0, :]) + 1
        if horizontal_lines_linewidth is None:
            ax1.hlines(horizontal_lines, line_beg_x, line_end_x, color=horizontal_lines_colors, linewidth=2,
                       linestyles=horizontal_lines_sytle)
        else:
            ax1.hlines(horizontal_lines, line_beg_x, line_end_x, color=horizontal_lines_colors,
                       linewidth=horizontal_lines_linewidth,
                       linestyles=horizontal_lines_sytle)

    if vertical_lines is not None:
        line_beg_y = 0
        line_end_y = len(spike_nums) - 1
        ax1.vlines(vertical_lines, line_beg_y, line_end_y, color=vertical_lines_colors,
                   linewidth=vertical_lines_linewidth,
                   linestyles=vertical_lines_sytle)

    if spike_train_format:
        ax1.set_xlim(min_time - 1, max_time + 1)
    else:
        ax1.set_xlim(-1, len(spike_nums[0, :]) + 1)
    # ax1.margins(x=0, tight=True)

    ax1.get_xaxis().set_visible(False)

    if title is None:
        ax1.set_title('Spikes raster plot')
    else:
        ax1.set_title(title)
    # Give x axis label for the spike raster plot
    # ax.xlabel('Frames')
    # Give y axis label for the spike raster plot
    ax1.set_ylabel(ylabel)

    if (axes_list is not None) and without_activity_sum:
        return

    # ################################################################################################
    # ################################ Activity sum plot part ################################
    # ################################################################################################
    if sliding_window_duration >= 1:
        # print("sliding_window_duration > 1")
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
            if debug_mode:
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
        if debug_mode:
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
    if (span_area_coords is not None) and (not span_area_only_on_raster):
        for index, span_area_coord in enumerate(span_area_coords):
            for coord in span_area_coord:
                if span_area_colors is not None:
                    color = span_area_colors[index]
                else:
                    color = "lightgrey"
                ax2.axvspan(coord[0], coord[1], alpha=0.5, facecolor=color, zorder=1)

    # early born
    if cells_to_highlight is not None and color_peaks_activity:
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
    if SCE_times is not None:
        ax_top = ax2.twiny()
        ax_top.set_frame_on(False)
        if spike_train_format:
            ax_top.set_xlim(min_time - 1, max_time + 1)
        else:
            ax_top.set_xlim(-1, len(spike_nums[0, :]) + 1)
        xticks_pos = []
        for times_tuple in SCE_times:
            xticks_pos.append(times_tuple[0])
        ax_top.set_xticks(xticks_pos)
        ax_top.xaxis.set_ticks_position('none')
        ax_top.set_xticklabels(np.arange(len(SCE_times)))
        plt.setp(ax_top.xaxis.get_majorticklabels(), rotation=90)
        if len(SCE_times) > 30:
            ax_top.xaxis.set_tick_params(labelsize=3)
        elif len(SCE_times) > 50:
            ax_top.xaxis.set_tick_params(labelsize=2)
        elif len(SCE_times) > 100:
            ax_top.xaxis.set_tick_params(labelsize=1)
        elif len(SCE_times) > 300:
            ax_top.xaxis.set_tick_params(labelsize=0.5)
        else:
            ax_top.xaxis.set_tick_params(labelsize=4)
    # print(f"max sum_spikes {np.max(sum_spikes)}, mean  {np.mean(sum_spikes)}, median {np.median(sum_spikes)}")
    ax2.set_ylim(0, np.max(sum_spikes))

    # color bar section
    if plot_with_amplitude:
        inner_2 = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                   subplot_spec=outer[1])  # , wspace=0.1, hspace=0.1)
        ax3 = fig.add_subplot(inner_2[0])  # plt.Subplot(fig, inner_2[0])
        fig.colorbar(scalar_map, cax=ax3)
    if axes_list is None:
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


def plot_sum_active_clusters(clusters_activations,
                             data_str, sliding_window_duration=None, show_fig=False, save_plot=True, axes_list=None,
                             fig_to_use=None, param=None,
                             save_formats="pdf"):
    if axes_list is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1, sharex=False,
                                gridspec_kw={'height_ratios': [1], 'width_ratios': [1]},
                                figsize=(20, 5))
        plt.tight_layout(pad=3, w_pad=7, h_pad=3)
    else:
        ax1 = axes_list[0]
        fig = fig_to_use

    n_clusters = len(clusters_activations)
    n_times = len(clusters_activations[0, :])
    if sliding_window_duration is None:
        sum_clusters = np.sum(clusters_activations, axis=0)
    else:
        sum_clusters = np.zeros(n_times)
        for t in np.arange(0, (n_times - sliding_window_duration)):
            # One spike by cell max in the sum process
            sum_value = np.sum(clusters_activations[:, t:(t + sliding_window_duration)], axis=1)
            sum_clusters[t] = len(np.where(sum_value)[0])
        sum_clusters[(n_times - sliding_window_duration):] = len(np.where(sum_value)[0])

    # expressed in percentages
    sum_clusters = sum_clusters / n_clusters
    sum_clusters *= 100

    x_value = np.arange(n_times)

    ax1.fill_between(x_value, 0, sum_clusters, facecolor="black")

    ax1.set_ylim(0, np.max(sum_clusters))

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
