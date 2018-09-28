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
        # if (eb_bins is not None) and (sum_perc > threshold_perc):
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