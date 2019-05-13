import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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

def plot_hist_distribution(distribution_data, description, param, values_to_scatter=None,
xticks_labelsize=10, yticks_labelsize=10, x_label_font_size=15, y_label_font_size=15,
                           labels=None, scatter_shapes=None, colors=None, tight_x_range=False,
                           twice_more_bins=False, background_color="black", labels_color="white",
                           xlabel="", ylabel=None, path_results=None, save_formats="pdf",
                           ax_to_use=None, color_to_use=None):
    """
    Plot a distribution in the form of an histogram, with option for adding some scatter values
    :param distribution_data:
    :param description:
    :param param:
    :param values_to_scatter:
    :param labels:
    :param scatter_shapes:
    :param colors:
    :param tight_x_range:
    :param twice_more_bins:
    :param xlabel:
    :param ylabel:
    :param save_formats:
    :return:
    """
    distribution = np.array(distribution_data)
    if color_to_use is None:
        hist_color = "blue"
    else:
        hist_color = color_to_use
    edge_color = "white"
    if tight_x_range:
        max_range = np.max(distribution)
        min_range = np.min(distribution)
    else:
        max_range = 100
        min_range = 0
    weights = (np.ones_like(distribution) / (len(distribution))) * 100
    if ax_to_use is None:
        fig, ax1 = plt.subplots(nrows=1, ncols=1,
                                gridspec_kw={'height_ratios': [1]},
                                figsize=(12, 12))
        ax1.set_facecolor(background_color)
        fig.patch.set_facecolor(background_color)
    else:
        ax1 = ax_to_use
    bins = int(np.sqrt(len(distribution)))
    if twice_more_bins:
        bins *= 2
    hist_plt, edges_plt, patches_plt = ax1.hist(distribution, bins=bins, range=(min_range, max_range),
                                                facecolor=hist_color,
                                                edgecolor=edge_color,
                                                weights=weights, log=False, label=description)
    if values_to_scatter is not None:
        scatter_bins = np.ones(len(values_to_scatter), dtype="int16")
        scatter_bins *= -1

        for i, edge in enumerate(edges_plt):
            # print(f"i {i}, edge {edge}")
            if i >= len(hist_plt):
                # means that scatter left are on the edge of the last bin
                scatter_bins[scatter_bins == -1] = i - 1
                break

            if len(values_to_scatter[values_to_scatter <= edge]) > 0:
                if (i + 1) < len(edges_plt):
                    bool_list = values_to_scatter < edge  # edges_plt[i + 1]
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                new_i = max(0, i - 1)
                                scatter_bins[i_bool] = new_i
                else:
                    bool_list = values_to_scatter < edge
                    for i_bool, bool_value in enumerate(bool_list):
                        if bool_value:
                            if scatter_bins[i_bool] == -1:
                                scatter_bins[i_bool] = i

        decay = np.linspace(1.1, 1.15, len(values_to_scatter))
        for i, value_to_scatter in enumerate(values_to_scatter):
            if i < len(labels):
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20, label=labels[i])
            else:
                ax1.scatter(x=value_to_scatter, y=hist_plt[scatter_bins[i]] * decay[i], marker=scatter_shapes[i],
                            color=colors[i], s=60, zorder=20)
    ax1.legend()

    if tight_x_range:
        ax1.set_xlim(min_range, max_range)
    else:
        ax1.set_xlim(0, 100)
        xticks = np.arange(0, 110, 10)

        ax1.set_xticks(xticks)
        # sce clusters labels
        ax1.set_xticklabels(xticks)
    ax1.yaxis.set_tick_params(labelsize=xticks_labelsize)
    ax1.xaxis.set_tick_params(labelsize=yticks_labelsize)
    ax1.tick_params(axis='y', colors=labels_color)
    ax1.tick_params(axis='x', colors=labels_color)
    # TO remove the ticks but not the labels
    # ax1.xaxis.set_ticks_position('none')

    if ylabel is None:
        ax1.set_ylabel("Distribution (%)", fontsize=30, labelpad=20)
    else:
        ax1.set_ylabel(ylabel, fontsize=y_label_font_size, labelpad=20)
    ax1.set_xlabel(xlabel, fontsize=x_label_font_size, labelpad=20)

    ax1.xaxis.label.set_color(labels_color)
    ax1.yaxis.label.set_color(labels_color)

    # padding between ticks label and  label axis
    # ax1.tick_params(axis='both', which='major', pad=15)

    if ax_to_use is None:
        fig.tight_layout()
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        if path_results is None:
            path_results = param.path_results
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{description}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}",
                                facecolor=fig.get_facecolor())

        plt.close()

def plot_scatters(x_coords, y_coords, size_scatter=30, ax_to_use=None, color_to_use=None, legend_str="",
                  xlabel="", ylabel="", filename_option="",
                  save_formats="pdf"):
    if (color_to_use is None):
        color = "cornflowerblue"
    else:
        color = color_to_use

    if ax_to_use is None:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 20))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
    else:
        ax = ax_to_use

    ax.scatter(x_coords, y_coords, color=color, edgecolors="white", marker="o",
               s=size_scatter, zorder=1, alpha=0.7, label=legend_str)

    ax.xaxis.set_tick_params(labelsize=5)

    ax.tick_params(axis='y', colors="white")
    ax.tick_params(axis='x', colors="white")

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    # legend_elements = [Patch(facecolor=color,
    #                          edgecolor="white",
    #                          label=f"{legend_str}")]
    ax.legend() # handles=legend_elements

    #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
    # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
    if ax_to_use is None:
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/plot_scatter_{filename_option}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}",
                        facecolor=fig.get_facecolor())
        plt.close()


def time_correlation_graph(time_lags_list, correlation_list, time_lags_dict, correlation_dict,
                           n_cells, time_window,
                           data_id, param, plot_cell_numbers=False,
                           title_option="",
                           cells_groups=None, groups_colors=None, set_y_limit_to_max=True,
                           set_x_limit_to_max=True, xlabel=None, size_cells=100, size_cells_in_groups=240,
                           time_stamps_by_ms=0.01, ms_scale=200, save_formats="pdf",
                           show_percentiles=None, ax_to_use=None, color_to_use=None,
                           value_to_text_in_cell=None):
    # value_to_text_in_cell if not None, dict with key cell (int) and value a string to plot
    # ms_scale represents the space between each tick
    if ((cells_groups is not None) and (len(cells_groups) > 0)) or (color_to_use is None):
        default_cell_color = "grey"
    else:
        default_cell_color = color_to_use

    if ax_to_use is None:
        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 20))
        fig.patch.set_facecolor("black")
        ax.set_facecolor("black")
    else:
        ax = ax_to_use

    ax.scatter(time_lags_list, correlation_list, color=default_cell_color, marker="o",
               s=size_cells, zorder=1, alpha=0.5)

    if cells_groups is not None:
        for group_id, cells in enumerate(cells_groups):
            for cell in cells:
                if cell in time_lags_dict:
                    ax.scatter(time_lags_dict[cell], correlation_dict[cell], color=groups_colors[group_id],
                               marker="o",
                               s=size_cells_in_groups, zorder=10)
                # else:
                #     print(f"{data_id}: cell {cell} not in time-correlation graph")

    if plot_cell_numbers:
        for cell in np.arange(n_cells):
            if cell in time_lags_dict:
                text_str = str(cell)
                if value_to_text_in_cell is not None:
                    text_str = value_to_text_in_cell[cell]
                ax.text(x=time_lags_dict[cell], y=correlation_dict[cell], s=text_str,
                        ha='center', va="center", fontsize=3, fontweight='bold')

    xticks_pos = []
    # display a tick every 200 ms, time being in seconds
    times_for_s_scale = ms_scale * 0.001
    time_window_s = (time_window / time_stamps_by_ms) * 0.001
    xticklabels = []
    labels_in_s = np.arange(-time_window_s * 2, time_window_s * 2 + 1, times_for_s_scale)
    pos_range = np.arange(-time_window * 2, time_window * 2 + 1, ms_scale * time_stamps_by_ms)
    # print(f"max_value {max_value}")
    if set_x_limit_to_max:
        min_range_index = 0
        max_range_index = len(pos_range) - 1
    else:
        max_value = np.max((np.abs(np.min(time_lags_list)), np.abs(np.max(time_lags_list))))
        min_range_index = np.searchsorted(pos_range, -max_value, side='left') - 1
        max_range_index = np.searchsorted(pos_range, max_value, side='right')
    # print(f"min_range_index {min_range_index}, max_range_index {max_range_index}, pos_range {pos_range}")
    labels_in_s = labels_in_s[min_range_index:max_range_index + 5:5]
    for index_pos, pos in enumerate(pos_range[min_range_index:max_range_index + 5:5]):
        xticks_pos.append(pos)
        xticklabels.append(np.round(labels_in_s[index_pos], 1))
    if len(labels_in_s) > 20:
        ax.xaxis.set_tick_params(labelsize=2)
    else:
        ax.xaxis.set_tick_params(labelsize=5)

    # print(f"xticks_pos {xticks_pos}")
    # print(f"xticklabels {xticklabels}")
    ax.set_xticks(xticks_pos)
    ax.set_xticklabels(xticklabels)
    ax.tick_params(axis='y', colors="white")
    ax.tick_params(axis='x', colors="white")
    if set_x_limit_to_max:
        ax.set_xlim(-time_window * 2, (time_window * 2) + 1)
    else:
        ax.set_xlim(pos_range[min_range_index], pos_range[max_range_index])

    if show_percentiles is not None:
        for perc_value in show_percentiles:
            correlation_threshold = np.percentile(correlation_list, perc_value)
            if set_x_limit_to_max:
                start_x = -time_window * 2
                end_x =  (time_window * 2) + 1
            else:
                start_x = pos_range[min_range_index]
                end_x = pos_range[max_range_index]
            ax.hlines(correlation_threshold, start_x, end_x, color="white", linewidth=1, linestyles="dashed")

    if set_y_limit_to_max:
        ax.set_ylim(0, 1.1)

    if xlabel is None:
        ax.set_xlabel("Time lag (s)")
    else:
        ax.set_xlabel(xlabel)
    ax.set_ylabel("Correlation")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")

    legend_elements = [Patch(facecolor=default_cell_color,
                             edgecolor="white",
                             label=f"{data_id} {title_option}")]
    ax.legend(handles=legend_elements)

    # plt.title(f"Time-correlation graph {data_id} {title_option}")

    #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
    # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
    if ax_to_use is None:
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{data_id}_time-correlation-graph_{title_option}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}",
                    facecolor=fig.get_facecolor())
        plt.close()
