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
from pattern_discovery.seq_solver.markov_way import sort_it_and_plot_it
from sortedcontainers import SortedList, SortedDict


class CellAssembliesStruct:
    def __init__(self, cellsinpeak, n_clusters, sce_clusters_id, sce_clusters_labels, neurons_labels,
                 cluster_with_best_silhouette_score, param, sliding_window_duration,
                 n_surrogate_k_mean, SCE_times, data_id):
        # cellsinpeak shape (n_cells, n_sces)
        self.cellsinpeak = cellsinpeak
        self.param = param
        self.n_cells = np.shape(cellsinpeak)[0]
        self.n_sces = np.shape(cellsinpeak)[1]
        self.n_clusters = n_clusters
        self.n_surrogate_k_mean = n_surrogate_k_mean
        # len(sce_clusters_id) == n_clusters, give the cluster_id for each cluster of sce
        self.sce_clusters_id = sce_clusters_id
        # associate for each cluster_id a sce. len(sce_clusters_labels) == n_sces, and each value is a cluster_id
        self.sce_clusters_labels = sce_clusters_labels
        # give the number of clusters that gives the best silhouette score overall
        self.cluster_with_best_silhouette_score = cluster_with_best_silhouette_score
        self.SCE_times = SCE_times
        self.data_id = data_id

        # to be computed later
        self.cellsinpeak_ordered = None
        # list contains the nb of cells in each cell assemblie cluster
        self.n_cells_in_cell_assemblies_clusters = None
        self.n_cells_not_in_cell_assemblies = None
        # novel order to display cells organized by cell assembly cluster, last cells being the one without cell assembly
        self.cells_indices = None
        # new sce indices
        self.sce_indices = None
        # give the number of sce in the no-assembly-sce, single-assembly and multiple-assembly groups respectively
        self.n_sce_in_assembly = None
        self.n_cells_in_single_cell_assembly_sce_cl = None
        self.n_cells_in_multiple_cell_assembly_sce_cl = None
        self.SCE_times = None
        self.activity_threshold = None
        self.data_descr = None
        # neurons_labels are the label to be display for each cell, in the original order (before any clustering)
        self.neurons_labels = neurons_labels
        self.sliding_window_duration = sliding_window_duration
        # key will be the cell assembly index (such as displayed) and the value is a list of list of 2 elements (first and
        # last sce index of each sce cluster part of this assembly)
        # used to disply lines around clusters in the heatmap
        self.sces_in_cell_assemblies_clusters = None
        # key is a int representing a sce, and value a list representing the id of cell assemblies cluster
        self.cell_assemblies_cluster_of_multiple_ca_sce = None

    def save_data_on_file(self, n_clusters):
        file_name = f'{self.param.path_results}/{self.data_id}_{n_clusters}_clusters_cell_assemblies_data_{self.param.time_str}.txt'

        with open(file_name, "w", encoding='UTF-8') as file:
            # first saving params
            file.write(f"#PARAM#" + '\n')
            file.write(f"data_id {self.data_id}" + '\n')
            file.write(f"n_clusters_for_kmean {self.n_clusters}" + '\n')
            file.write(f"n_surrogate_k_mean {self.n_surrogate_k_mean}" + '\n')
            file.write(f"activity_threshold {self.param.activity_threshold}" + '\n')

            file.write(f"#CELL_ASSEMBLIES#" + '\n')

            if len(self.n_cells_in_cell_assemblies_clusters) == 0:
                file.write(f"NONE" + '\n')
                return

            # SCA_SCE: single cell-assembly SCE
            start = 0
            for cluster_id, n_cells in enumerate(self.n_cells_in_cell_assemblies_clusters):
                stop = start + n_cells
                file.write(f"SCA_cluster:{cluster_id}:{' '.join(map(str, self.cells_indices[start:stop]))}" + '\n')
                start = stop

            # MCA_SCE: multiple cell-assembly SCE
            for sce_id, ca_ids in self.cell_assemblies_cluster_of_multiple_ca_sce.items():
                file.write(f"MCA_SCE:{self.SCE_times[sce_id][0]} {self.SCE_times[sce_id][1]}:")
                file.write(f"{' '.join(map(str, ca_ids))}" + '\n')
            # then if cell assemblies, write the times in frames (first and last of each SCE) by cell assemblies
            # first single cell assemblies, then multiple cell assemblies
            n_sces_not_in_ca = self.n_sce_in_assembly[0]
            n_sces_so_far = 0
            start_index = n_sces_not_in_ca + n_sces_so_far
            for ca_id, sces_indices_tuple in self.sces_in_cell_assemblies_clusters.items():
                n_sces = sces_indices_tuple[0][1] - sces_indices_tuple[0][0]
                last_index = start_index + n_sces
                # print(f"self.sce_indices {self.sce_indices}")
                # print(f"start_index {start_index}")
                # print(f"last_index {last_index}")
                file.write(f"single_sce_in_ca:{ca_id}:")
                for i, index_sce_period in enumerate(self.sce_indices[start_index:last_index]):
                    sce_period = self.SCE_times[int(index_sce_period)]
                    file.write(f"{sce_period[0]} {sce_period[1]}")
                    if i < len(self.sce_indices[start_index:last_index])-1:
                        file.write(f"#")
                file.write('\n')
                start_index += n_sces
            if len(self.cell_assemblies_cluster_of_multiple_ca_sce) > 0:
                sce_ids = np.array(list(self.cell_assemblies_cluster_of_multiple_ca_sce.keys()))
                file.write(f"multiple_sce_in_ca:")
                for i, index_sce_period in enumerate(sce_ids):
                    # print(f"index_sce_period {index_sce_period}")
                    # print(f"self.sce_indices[int(index_sce_period)] {self.sce_indices[int(index_sce_period)]}")
                    sce_period = self.SCE_times[index_sce_period]
                    file.write(f"{sce_period[0]} {sce_period[1]}")
                    if i < len(sce_ids)-1:
                        file.write(f"#")
                file.write('\n')

    def plot_cell_assemblies(self, data_descr, SCE_times, activity_threshold,
                             spike_nums, sce_times_bool=None,
                             display_only_cell_assemblies_on_raster=False,
                             with_cells_in_cluster_seq_sorted=False,
                             save_formats="pdf", show_fig=False):

        self.data_descr = data_descr
        self.SCE_times = SCE_times
        self.activity_threshold = activity_threshold
        background_color = "black"

        fig = plt.figure(figsize=(20, 14))
        fig.set_tight_layout({'rect': [0, 0, 1, 1], 'pad': 1, 'h_pad': 2})
        outer = gridspec.GridSpec(2, 1, height_ratios=[60, 40])

        inner_bottom = gridspec.GridSpecFromSubplotSpec(2, 1,
                                                        subplot_spec=outer[1], height_ratios=[10, 2])

        # clusters display
        inner_top = gridspec.GridSpecFromSubplotSpec(1, 1,
                                                     subplot_spec=outer[0])

        # top is bottom and bottom is top, so the raster is under
        # ax1 contains raster
        ax1 = fig.add_subplot(inner_bottom[0])
        # ax2 contains the peak activity diagram
        ax2 = fig.add_subplot(inner_bottom[1], sharex=ax1)

        ax3 = fig.add_subplot(inner_top[0])
        fig.patch.set_facecolor(background_color)

        self.plot_raster(axes_list=[ax1, ax2], spike_nums=spike_nums,
                         with_cells_in_cluster_seq_sorted=with_cells_in_cluster_seq_sorted,
                         sce_times_bool=sce_times_bool,
                         display_only_cell_assemblies_on_raster=display_only_cell_assemblies_on_raster)

        self.plot_cells_vs_sce(ax2=ax3)
        # show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
        #                          significant_sce_clusters=significant_sce_clusters[n_cluster],
        #                          n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
        #                          cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
        #                          data_str=data_descr, path_results=param.path_results,
        #                          show_silhouettes=True, neurons_labels=labels,
        #                          surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
        #                          axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats="pdf")
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        bonus_str = ""
        if display_only_cell_assemblies_on_raster:
            bonus_str = "_only_sca_"
        for save_format in save_formats:
            fig.savefig(f'{self.param.path_results}/{self.data_descr}_{self.n_clusters}_'
                        f'cell_assemblies{bonus_str}.{save_format}',
                    format=f"{save_format}", facecolor=fig.get_facecolor())
        if show_fig:
            plt.show()
        plt.close()

    def plot_raster(self, axes_list, spike_nums, with_cells_in_cluster_seq_sorted=False, sce_times_bool=None,
                    display_only_cell_assemblies_on_raster=False):
        # this section will order the spike_nums for display purpose
        clustered_spike_nums = np.copy(spike_nums)

        cell_labels = []
        for index in self.cells_indices:
            cell_labels.append(self.neurons_labels[index])
        cluster_horizontal_thresholds = []
        cells_to_highlight = []
        cells_to_highlight_colors = []
        start = 0

        clustered_spike_nums = clustered_spike_nums[self.cells_indices, :]

        cells_group_numbers = []
        if len(self.n_cells_in_cell_assemblies_clusters) > 0:
            cells_group_numbers.extend(self.n_cells_in_cell_assemblies_clusters)
        if self.n_cells_not_in_cell_assemblies > 0:
            cells_group_numbers.append(self.n_cells_not_in_cell_assemblies)

        seq_dict = dict()
        colors_for_seq_dict = dict()
        for group_number, group_size in enumerate(cells_group_numbers):
            if with_cells_in_cluster_seq_sorted and (group_size > 4):
                # TODO: use code from markov_way with surrogates generation, need to be more modulable
                self.param.error_rate = 0.25
                self.param.max_branches = 10
                self.param.time_inter_seq = 50
                self.param.min_duration_intra_seq = 0
                self.param.min_len_seq = 5
                self.param.min_rep_nb = 3
                # link_seq_categories = significant_category_dict,
                # link_seq_color = colors_for_seq_list,
                # link_seq_line_width = 0.8,
                # link_seq_alpha = 0.9,
                to_sort = clustered_spike_nums[start:start + group_size, :]
                best_seq, seq_dict_tmp = sort_it_and_plot_it(spike_nums=to_sort, param=self.param,
                                                             sce_times_bool=sce_times_bool,
                                                             sliding_window_duration=self.sliding_window_duration,
                                                             activity_threshold=self.param.activity_threshold,
                                                             use_only_uniformity_method=True,
                                                             save_plots=False)

                # if a list of ordered_indices, the size of the list is equals to ne number of cells,
                # each list correspond to the best order with this cell as the first one in the ordered seq
                if best_seq is not None:
                    clustered_spike_nums[start:start + group_size, :] = to_sort[best_seq, :]
                    color = cm.nipy_spectral(float(group_number + 2) /
                                             (len(self.n_cells_in_cell_assemblies_clusters) + 1))
                    for seq, times in seq_dict_tmp.items():
                        # updating the cell number to correspond to the whole raster
                        new_seq = tuple([cell+start for cell in seq])
                        seq_dict[new_seq] = times
                        colors_for_seq_dict[new_seq] = color

                    real_data_result_for_stat = SortedDict()
                    for key, value in seq_dict_tmp.items():
                        # print(f"len: {len(key)}, seq: {key}, rep: {len(value)}")
                        if len(key) not in real_data_result_for_stat:
                            real_data_result_for_stat[len(key)] = dict()
                            real_data_result_for_stat[len(key)]["rep"] = []
                            real_data_result_for_stat[len(key)]["duration"] = []
                        real_data_result_for_stat[len(key)]["rep"].append(len(value))
                        list_of_durations = []
                        # keeping the duration of each repetition
                        for time_stamps in value:
                            list_of_durations.append(time_stamps[-1] - time_stamps[0])
                        real_data_result_for_stat[len(key)]["duration"].append(list_of_durations)
                    results_dict = real_data_result_for_stat
                    file_name = f'{self.param.path_results}/{self.data_descr}_' \
                                f'{self.n_clusters}_{group_number}_cell_assemblies_sorting_results_{self.param.time_str}.txt'

                    min_len = 1000
                    max_len = 0
                    with open(file_name, "w", encoding='UTF-8') as file:
                        for key in results_dict.keys():
                            min_len = np.min((key, min_len))
                            max_len = np.max((key, max_len))

                        # key reprensents the length of a seq
                        for key in np.arange(min_len, max_len + 1):
                            nb_rep_seq = None
                            flat_durations = None
                            if key in results_dict:
                                nb_rep_seq = results_dict[key]["rep"]
                                durations = results_dict[key]["duration"]
                                flat_durations = [item for sublist in durations for item in sublist]

                            str_to_write = ""
                            str_to_write += f"### Length: {key} cells \n"
                            if (nb_rep_seq is not None) and (len(nb_rep_seq) > 0):

                                str_to_write += f"# Real data (nb seq: {len(nb_rep_seq)}), " \
                                                f"repetition: mean {np.round(np.mean(nb_rep_seq), 3)}"
                                if np.std(nb_rep_seq) > 0:
                                    str_to_write += f", std {np.round(np.std(nb_rep_seq), 3)}"
                                str_to_write += f"#, duration: " \
                                                f": mean {np.round(np.mean(flat_durations), 3)}"
                                if np.std(flat_durations) > 0:
                                    str_to_write += f", std {np.round(np.std(flat_durations), 3)}"
                                str_to_write += f"\n"
                            str_to_write += '\n'
                            str_to_write += '\n'
                            file.write(f"{str_to_write}")

            if (self.n_cells_not_in_cell_assemblies > 0) and (group_number == (len(cells_group_numbers) - 1)):
                continue
            if len(self.n_cells_in_cell_assemblies_clusters) == 0:
                continue

            # coloring cell assemblies
            color = cm.nipy_spectral(float(group_number + 1) / (len(self.n_cells_in_cell_assemblies_clusters) + 1))
            cell_indices_to_color = list(np.arange(start, start + group_size))
            cells_to_highlight.extend(cell_indices_to_color)
            cells_to_highlight_colors.extend([color] * len(cell_indices_to_color))

            start += group_size

            if group_number < (len(cells_group_numbers) - 1):
                cluster_horizontal_thresholds.append(start)

        if len(cell_labels) > 100:
            y_ticks_labels_size = 1
        else:
            y_ticks_labels_size = 3
        spike_shape_size = 1
        if len(cell_labels) > 150:
            spike_shape_size = 0.5
        if len(cell_labels) > 500:
            spike_shape_size = 0.2

        seq_times_to_color_dict = None
        if with_cells_in_cluster_seq_sorted and len(seq_dict) > 0:
            seq_times_to_color_dict = seq_dict

        if display_only_cell_assemblies_on_raster:
            n_cells = len(clustered_spike_nums)
            if (self.n_cells_not_in_cell_assemblies > 0) and (self.n_cells_not_in_cell_assemblies < n_cells):
                n_cells_to_keep = n_cells - int(self.n_cells_not_in_cell_assemblies)
                clustered_spike_nums = clustered_spike_nums[:n_cells_to_keep]
                cell_labels = cell_labels[:n_cells_to_keep]

        colors_for_seq_list = ["blue", "red", "limegreen", "grey", "orange", "cornflowerblue", "yellow", "seagreen",
                               "magenta"]
        plot_spikes_raster(spike_nums=clustered_spike_nums, param=self.param,
                           spike_train_format=False,
                           title="",
                           file_name=f"cell assemblies raster plot_{self.data_descr}_{self.n_clusters}_clusters",
                           y_ticks_labels=cell_labels,
                           y_ticks_labels_size=y_ticks_labels_size,
                           y_ticks_labels_color="white",
                           x_ticks_labels_color="white",
                           activity_sum_plot_color="white",
                           activity_sum_face_color="black",
                           without_ticks=True,
                           save_raster=False,
                           show_raster=False,
                           plot_with_amplitude=False,
                           activity_threshold=self.activity_threshold,
                           raster_face_color='black',
                           cell_spikes_color='white',
                           horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                           horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                           horizontal_lines_sytle="dashed",
                           horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                           span_area_coords=[self.SCE_times],
                           span_area_colors=['white'],
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           seq_times_to_color_dict=seq_times_to_color_dict,
                           link_seq_color=colors_for_seq_list, #colors_for_seq_dict,
                           link_seq_line_width=0.6,
                           link_seq_alpha=0.9,
                           jitter_links_range=5,
                           min_len_links_seq=3,
                           sliding_window_duration=self.sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="o",
                           spike_shape_size=spike_shape_size,
                           save_formats="pdf",
                           axes_list=axes_list,
                           SCE_times=self.SCE_times)

    def plot_cells_vs_sce(self, ax2):
        # ############################
        # Heatmap cells vs SCE
        # ############################

        n_cell_assemblies = len(self.n_cells_in_cell_assemblies_clusters)

        cluster_horizontal_thresholds = []
        nb_cells_by_cell_ass_cluster_y_coord = []
        # clusters labels
        nb_cells_by_cell_ass_cluster = []
        start = 0

        cells_group_numbers = []
        if n_cell_assemblies > 0:
            cells_group_numbers.extend(self.n_cells_in_cell_assemblies_clusters)
        if self.n_cells_not_in_cell_assemblies > 0:
            cells_group_numbers.append(self.n_cells_not_in_cell_assemblies)

        for cell_group_id, group_size in enumerate(cells_group_numbers):
            # print(f"cell_group_id {cell_group_id}, group_size {group_size}")
            if (self.n_cells_not_in_cell_assemblies > 0) and (cell_group_id == (len(cells_group_numbers) - 1)):
                nb_cells_by_cell_ass_cluster_y_coord.append((start + self.n_cells) / 2)
                nb_cells_by_cell_ass_cluster.append(self.n_cells_not_in_cell_assemblies)
                continue
            if len(self.n_cells_in_cell_assemblies_clusters) == 0:
                continue

            nb_cells_by_cell_ass_cluster_y_coord.append(start + (group_size / 2))
            nb_cells_by_cell_ass_cluster.append(int(group_size))

            range_group = np.arange(start, start+group_size)
            for cell_id in range_group:
                spikes_index = np.where(self.cellsinpeak_ordered[cell_id, :])[0]
                # print(f"spikes_index {spikes_index}, self.n_sce_in_assembly[0] {self.n_sce_in_assembly[0]}")
                # keeping only spikes that are part of sce belonging to cell assemblies
                spikes_index = spikes_index[spikes_index >= self.n_sce_in_assembly[0]]
                # print(f"spikes_index filtered {spikes_index}")
                if len(spikes_index) > 0:
                    # print("you shall not pass")
                    # K +2 to avoid zero and one, and at the end we will substract 2
                    self.cellsinpeak_ordered[cell_id, spikes_index] = cell_group_id + 2

            start += group_size

            if cell_group_id < (len(cells_group_numbers) - 1):
                cluster_horizontal_thresholds.append(start)

        # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
        # value to one represent the cells spikes without assembly, then number 2 represent the cell assembly 0, etc...
        if np.max(self.cellsinpeak_ordered) <= 1:
            # it means that no sce cluster is significant
            list_color = ['black', 'white']
            bounds = [-0.5, 0.5, 1.5]
        else:
            self.cellsinpeak_ordered = self.cellsinpeak_ordered - 2
            # print(f"self.cellsinpeak_ordered {self.cellsinpeak_ordered}")
            # print(f"np.min(ordered_n_cells_in_peak) {np.min(ordered_n_cells_in_peak)}")
            list_color = ['black', 'white']
            bounds = [-2.5, -1.5, -0.5]
            # bounds = [-1.5, 0.5, 1.5]
            for i in np.arange(n_cell_assemblies):
                color = cm.nipy_spectral(float(i + 1) / (n_cell_assemblies + 1))
                list_color.append(color)
                bounds.append(i + 0.5)

        cmap = mpl.colors.ListedColormap(list_color)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # rasterized=True used to remove the grid
        sns.heatmap(self.cellsinpeak_ordered, cbar=False, ax=ax2, cmap=cmap, norm=norm, rasterized=True)

        if self.neurons_labels is not None:
            ordered_neurons_labels = []
            for index in self.cells_indices:
                ordered_neurons_labels.append(self.neurons_labels[index])
            ax2.set_yticks(np.arange(len(ordered_neurons_labels)) + 0.5)
            ax2.set_yticklabels(ordered_neurons_labels)
        else:
            ax2.set_yticks(np.arange(self.n_cells) + 0.5)
            ax2.set_yticklabels(self.cells_indices.astype(int))

        if self.n_cells > 100:
            ax2.yaxis.set_tick_params(labelsize=3)
        elif self.n_cells > 200:
            ax2.yaxis.set_tick_params(labelsize=2)
        elif self.n_cells > 400:
            ax2.yaxis.set_tick_params(labelsize=1)
        else:
            ax2.yaxis.set_tick_params(labelsize=4)

        # creating axis at the top
        # ax_top = ax2.twiny()
        ax_right = ax2.twinx()
        # ax2.set_frame_on(False)

        # ax_top.set_frame_on(False)
        # ax_top.set_xlim((0, self.n_sces))
        # ax_top.set_xticks(cluster_sce_x_ticks_coord)
        # sce clusters labels
        # ax_top.set_xticklabels(significant_sce_clusters)

        # print(f"nb_cells_by_cluster_of_cells_y_coord {nb_cells_by_cluster_of_cells_y_coord} "
        #       f"nb_cells_by_cluster_of_cells {nb_cells_by_cluster_of_cells}")

        # ax_right.set_frame_on(False)

        ax_right.set_ylim((0, self.n_cells))
        ax_right.set_yticks(nb_cells_by_cell_ass_cluster_y_coord)
        ax_right.tick_params(axis='both', which='both', length=0)
        # clusters labels
        ax_right.set_yticklabels(nb_cells_by_cell_ass_cluster, fontweight="bold")
        ax_right.yaxis.set_tick_params(labelsize=14)
        for tick_label in ax_right.get_yticklabels():
            tick_label.set_color("white")

        ax2.set_xticks(np.arange(self.n_sces) + 0.5)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        ax2.tick_params(axis='both', which='both', length=0)
        # sce labels
        ax2.set_xticklabels(self.sce_indices.astype(int))
        if self.n_sces > 100:
            ax2.xaxis.set_tick_params(labelsize=3)
        elif self.n_sces > 200:
            ax2.xaxis.set_tick_params(labelsize=2)
        elif self.n_sces > 400:
            ax2.xaxis.set_tick_params(labelsize=1)
        else:
            ax2.xaxis.set_tick_params(labelsize=4)
        # horizontal lines showing the border between cell assemblies clusters, not so necessary as they are colored
        # ax2.hlines(cluster_horizontal_thresholds, self.n_sce_in_assembly[0], self.n_sces,
        #            color="white", linewidth=1,
        #            linestyles="dashed")
        # put a line between sce with one single cell assembly and those with multiple
        ax2.vlines([self.n_sce_in_assembly[0]+self.n_sce_in_assembly[1]], 0,
                   self.n_cells - self.n_cells_not_in_cell_assemblies, color="white", linewidth=2,
                   linestyles="dashed")
        start = 0
        for index, group_size in enumerate(self.n_cells_in_cell_assemblies_clusters):
            if index not in self.sces_in_cell_assemblies_clusters:
                # then there is no significant sce cluster
                start += group_size
                continue
            sces_clusters_borders = self.sces_in_cell_assemblies_clusters[index]
            for sces_borders in sces_clusters_borders:
                y_bottom = start
                y_top = start + group_size
                x_left = sces_borders[0]
                x_right = sces_borders[1]
                linewidth = 2
                color_border = "white"
                ax2.vlines(x_left, y_bottom, y_top, color=color_border, linewidth=linewidth)
                ax2.vlines(x_right, y_bottom, y_top, color=color_border, linewidth=linewidth)
                ax2.hlines(y_bottom, x_left, x_right, color=color_border, linewidth=linewidth)
                ax2.hlines(y_top, x_left, x_right, color=color_border, linewidth=linewidth)
            start += group_size

        # plt.setp(ax2.xaxis.get_majorticklabels(), rotation=90)
        plt.setp(ax2.yaxis.get_majorticklabels(), rotation=0)
        ax2.tick_params(axis='y', colors="white")
        ax2.tick_params(axis='x', colors="white")
        start = 0
        for i, cell_group_size in enumerate(self.n_cells_in_cell_assemblies_clusters):
            color = cm.nipy_spectral(float(i + 1) / (n_cell_assemblies + 1))
            for index in np.arange(start, (start+cell_group_size)):
                ax2.get_yticklabels()[index].set_color(color)
            start += cell_group_size

        ax2.invert_yaxis()


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


def surrogate_clustering(m_sces, n_clusters, n_surrogate, n_trials, perc_threshold,
                         fct_to_keep_best_silhouettes, debug_mode=False):
    """

    :param m_sces: sce matrix used for the clustering after permutation
    :param n_clusters: number of clusters
    :param n_surrogate: number of surrogates
    :param n_trials: number of trials by surrogate, keeping one avg silhouette by surrogate
    :param perc_threshold: threshold as percentile (int)
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
        best_silhouettes_clusters_avg = np.zeros(n_clusters)
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
        # print(f"surrogate_silhouettes {surrogate_silhouettes}, perc_threshold {perc_threshold}")
    percentile_result = np.percentile(surrogate_silhouettes, perc_threshold)
    return percentile_result


def clusters_on_sce_from_covnorm(cells_in_sce, range_n_clusters, fct_to_keep_best_silhouettes=np.mean,
                                 n_surrogate=1000, neurons_labels=None,
                                 perc_threshold_for_surrogate=95, debug_mode=False):
    """

    :param cells_in_sce:
    :param range_n_clusters:
    :param fct_to_keep_best_silhouettes: function used to keep the best trial, will be applied on all the silhouette
    scores of each trials, the max will be kept
    :return:
    """

    m_sces = cells_in_sce
    # normalized covariance matrix
    m_sces = covnorm(m_sces)
    # m_sces = np.corrcoef(m_sces)

    # key is the nth clusters as int, value is a list of list of SCE
    # (each list representing a cluster, so we have as many list as the number of cluster wanted)
    dict_best_clusters = dict()
    for i in range_n_clusters:
        dict_best_clusters[i] = []

    # nb of time to apply one given number of cluster
    n_trials = 100
    best_kmeans_by_cluster = dict()
    surrogate_percentiles_by_n_cluster = dict()
    # will keep the best silhouette score (depending on the fct_to_keep_best_silhouettes) for each number of cluster
    best_silhouette_score_by_n_cluster = np.zeros(np.max(range_n_clusters)+1)
    for n_clusters in range_n_clusters:
        # if debug_mode:
        print(f"kmeans {n_clusters} clusters")

        surrogate_percentile = surrogate_clustering(m_sces=m_sces, n_clusters=n_clusters,
                                                    n_surrogate=n_surrogate,
                                                    n_trials=100,
                                                    fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                                                    perc_threshold=perc_threshold_for_surrogate, debug_mode=True)

        best_kmeans = None
        best_silhouettes = 0
        for trial in np.arange(n_trials):
            kmeans = KMeans(n_clusters=n_clusters).fit(m_sces)
            cluster_labels = kmeans.labels_
            sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')
            local_clusters_silhouette = np.zeros(n_clusters)
            for i in range(n_clusters):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = \
                    sample_silhouette_values[cluster_labels == i]
                avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
                local_clusters_silhouette[i] = avg_ith_cluster_silhouette_values

            # compute a score based on the silhouette of each cluster for this trial and compare it with the best score
            # so far, keeping it if it's better
            computed_score = fct_to_keep_best_silhouettes(local_clusters_silhouette)
            if computed_score > best_silhouettes:
                best_silhouettes = computed_score
                best_silhouette_score_by_n_cluster[n_clusters] = computed_score
                best_kmeans = kmeans

        best_kmeans_by_cluster[n_clusters] = best_kmeans
        surrogate_percentiles_by_n_cluster[n_clusters] = surrogate_percentile

        print(f"best avg silh: {best_silhouette_score_by_n_cluster[n_clusters]}, surr: {surrogate_percentile}")

    return best_kmeans_by_cluster, m_sces, surrogate_percentiles_by_n_cluster, \
           np.argmax(best_silhouette_score_by_n_cluster)


def give_significant_sce_clusters(kmeans_dict, range_n_clusters, m_sces, surrogate_percentile):
    significant_sce_clusters = dict()
    for n_cluster in range_n_clusters:
        if kmeans_dict[n_cluster] is None:
            continue
        cluster_labels = kmeans_dict[n_cluster].labels_
        significant_sce_clusters[n_cluster] = []

        # keeping only the significant clusters
        sample_silhouette_values = metrics.silhouette_samples(m_sces, cluster_labels, metric='euclidean')

        for cluster_id in range(n_cluster):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == cluster_id]
            avg_ith_cluster_silhouette_values = np.mean(ith_cluster_silhouette_values)
            if avg_ith_cluster_silhouette_values > surrogate_percentile[n_cluster]:
                # then we keep it
                significant_sce_clusters[n_cluster].append(cluster_id)
        significant_sce_clusters[n_cluster] = np.array(significant_sce_clusters[n_cluster])

    return significant_sce_clusters


def plot_silhouettes(ax0, kmeans, m_sces, n_clusters, surrogate_silhouette_avg):
    cluster_labels = kmeans.labels_
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
                          facecolor=color, edgecolor=color, alpha=1)

        # putting a white edge around significan silhouette
        if surrogate_silhouette_avg is not None:
            if np.mean(ith_cluster_silhouette_values) > surrogate_silhouette_avg:
                ax0.plot(ith_cluster_silhouette_values, np.arange(y_lower, y_upper),
                         color="white", linewidth=1)
                # horizontal white line
                ax0.hlines(y_upper - 1, 0, ith_cluster_silhouette_values[-1],
                           color="white", linewidth=1)

        # Label the silhouette plots with their cluster numbers at the middle
        ax0.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i), color="white")

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax0.set_title("The silhouette plot for the various clusters.")
    ax0.set_xlabel("The silhouette coefficient values")
    ax0.set_ylabel("Cluster label")

    silhouette_avg = metrics.silhouette_score(m_sces, cluster_labels, metric='euclidean')
    if surrogate_silhouette_avg is not None:
        ax0.axvline(x=surrogate_silhouette_avg, color="white", linestyle="--")
    # The vertical line for average silhouette score of all the values
    ax0.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax0.set_yticks([])  # Clear the yaxis labels / ticks
    ax0.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])


def plot_covnorm_matrix(ax1, m_sces, n_clusters, cluster_labels):
    # display the normlized covariance matrix organized by cluster of SCE such as detected by initial kmeans
    # contains the neurons from the SCE, but ordered by cluster
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


# TODO: do shuffling before the real cluster
# TODO: when showing co-var, show the 95th percentile of shuffling, to see which cluster is significant

def show_co_var_first_matrix(cells_in_peak, m_sces, n_clusters, kmeans, cluster_labels_for_neurons,
                             significant_sce_clusters,
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

    # ################
    # silhouettes plot
    # ################
    if show_silhouettes:
        # Compute the silhouette scores for each sample
        plot_silhouettes(ax0=ax0, kmeans=kmeans, m_sces=m_sces, n_clusters=n_clusters,
                         surrogate_silhouette_avg=surrogate_silhouette_avg)

    # ############################
    # Normalized covariance plot
    # ############################
    plot_covnorm_matrix(ax1=ax1, m_sces=m_sces, n_clusters=n_clusters, cluster_labels=cluster_labels)

    # ############################
    # Heatmap cells vs SCE
    # ############################

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
    # first we put the sce that are not significant
    non_significant_sce_cluster = np.setdiff1d(np.arange(n_clusters), significant_sce_clusters)
    for sce_cluster_id in non_significant_sce_cluster:
        e = np.equal(cluster_labels, sce_cluster_id)
        nb_k = np.sum(e)
        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        sce_indices_for_each_clusters[sce_cluster_id] = np.arange(start, start + nb_k)
        old_pos = np.where(e)[0]
        for i, sce_index in enumerate(np.arange(start, start + nb_k)):
            new_sce_labels[sce_index] = old_pos[i]
        start += nb_k

    start_coord_significant_sce_clusters = start

    for index_k, k in enumerate(significant_sce_clusters):
        e = np.equal(cluster_labels, k)
        nb_k = np.sum(e)
        if color_each_clusters:
            for cell in np.arange(len(cells_in_peak)):
                spikes_index = np.where(cells_in_peak[cell, e] == 1)[0]
                to_put_to_true = np.where(e)[0][spikes_index]
                tmp_e = np.zeros(len(e), dtype="bool")
                tmp_e[to_put_to_true] = True
                # K +2 to avoid zero and one, and at the end we will substract 1
                cells_in_peak[cell, tmp_e] = k + 2

        ordered_cells_in_peak[:, start:start + nb_k] = cells_in_peak[:, e]
        sce_indices_for_each_clusters[k] = np.arange(start, start + nb_k)
        old_pos = np.where(e)[0]
        for i, sce_index in enumerate(np.arange(start, start + nb_k)):
            new_sce_labels[sce_index] = old_pos[i]
        start += nb_k

        if (index_k + 1) < len(significant_sce_clusters):
            if index_k == 0:
                cluster_x_ticks_coord.append((start + start_coord_significant_sce_clusters) / 2)
            else:
                cluster_x_ticks_coord.append((start + cluster_vertical_thresholds[-1]) / 2)
            cluster_vertical_thresholds.append(start)
        else:
            if len(cluster_vertical_thresholds) == 0:
                cluster_x_ticks_coord.append((start + start_coord_significant_sce_clusters) / 2)
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
        list_color = ['black', 'white']
        bounds = [-2.5, -1.5, -0.5]
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
    # rasterized=True used to remove the grid
    sns.heatmap(ordered_n_cells_in_peak, cbar=False, ax=ax2, cmap=cmap, norm=norm, rasterized=True)
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
    ax_top.set_xticklabels(significant_sce_clusters)

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
    ax2.hlines(cluster_horizontal_thresholds, start_coord_significant_sce_clusters, np.shape(cells_in_peak)[1],
               color="red", linewidth=1,
               linestyles="dashed")
    ax2.vlines([start_coord_significant_sce_clusters] + cluster_vertical_thresholds, 0,
               np.shape(cells_in_peak)[0], color="red", linewidth=1,
               linestyles="dashed")
    # print(f"n_clusters {n_clusters}, cluster_vertical_thresholds {cluster_vertical_thresholds}")
    for index, cluster in enumerate(significant_sce_clusters):
        if cluster not in clusters_coords_dict:
            # print(f"cluster {cluster} with no cells")
            # means no cell has this cluster as main cluster
            continue
        y_bottom, y_top = clusters_coords_dict[cluster]
        x_left = start_coord_significant_sce_clusters if (index == 0) else cluster_vertical_thresholds[index - 1]
        x_right = np.shape(cells_in_peak)[1] if (index == (len(significant_sce_clusters) - 1)) \
            else cluster_vertical_thresholds[index]
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
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{path_results}/{data_str}_{n_clusters}_sce_clusters.{save_format}',
                    format=f"{save_format}")
    if show_fig:
        plt.show()
        plt.close()


def find_cluster_labels_for_neurons(cells_in_peak, cluster_labels, m_sces, significant_clusters):
    """
    Building cell_assemblies
    :param cells_in_peak:
    :param cluster_labels:
    :param m_sces:
    :param significant_clusters:
    :return: cluster_labels_for_neurons np.array of len n_cells, each value correpond to the cell assembly
    the cell belongs, if the value equal -1, then the cell doesn't belong to any cell assemblie
    """
    cluster_labels_for_neurons = np.zeros(np.shape(cells_in_peak)[0], dtype="int8")
    # sorting neurons spikes, keeping them only in one cluster, the one with the most percentage spikes from this neuron
    # only from sce clusters significant
    # if spikes < 2 in any clusters, then removing spikes
    # going neuron by neuron,

    for n, events in enumerate(cells_in_peak):
        if len(significant_clusters) == 0:
            cluster_labels_for_neurons[n] = -1
            continue
        pos_events = np.where(events)[0]
        # will contains the number of spike in each sce for the neuron n
        max_clusters = np.zeros(len(significant_clusters), dtype="float")
        for p in pos_events:
            # p correspond to the index of one SCE
            sce_cluster_of_p = cluster_labels[p]
            if sce_cluster_of_p in significant_clusters:
                index = np.where(significant_clusters == sce_cluster_of_p)[0][0]
                max_clusters[index] += 1
        if np.max(max_clusters) < 2:
            # it should spike at least 2 times in a SCE cluster to be part of it
            cluster_labels_for_neurons[n] = -1
        else:
            # putting max_clusters into percentages, it means what percentages of sce the cell is spiking on
            max_clusters_perc = np.copy(max_clusters)
            for i in np.arange(len(max_clusters)):
                # i is the index of a sce cluster in significant_clusters
                nb_sce_in_cluster = len(np.where(cluster_labels == significant_clusters[i])[0])
                max_clusters_perc[i] = (max_clusters[i] / nb_sce_in_cluster) * 100
            # selecting the cluster with the most spikes from neuron n
            max_value = np.max(max_clusters_perc)
            pos_max_value = np.where(max_clusters_perc == max_value)[0]
            if len(pos_max_value) > 1:
                # if more than one sce with the max, we keep the one with the more spikes in it
                max_spikes = 0
                best_pos = 0
                for pos in pos_max_value:
                    if max_spikes < max_clusters[pos]:
                        max_spikes = max_clusters[pos]
                        best_pos = pos
                pos_max_value = best_pos
            else:
                pos_max_value = pos_max_value[0]
            # max_cluster = np.argmax(max_clusters_perc)
            cluster_labels_for_neurons[n] = significant_clusters[pos_max_value]
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
    file_name = f'{param.path_results}/{data_descr}_{n_cluster}_clusters_stat_k_mean_v_{param.time_str}.txt'
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


def statistical_cell_assemblies_def(cell_assemblies_struct,
                                    n_surrogate=1000, debug_mode=False):
    """
    :param sce_clusters_id:
    :param sce_clusters_labels:
    :param cellsinpeak:
    :return: will modify the cell_assemblies_struct instance and update the cellsinpeak_ordered and other useful
    variables used for display purpose
    """
    """
    Each cluster of SCE was then associated to a cell assembly which comprised those cells that significantly 
    participated to the SCE events within that particular cluster. 
    Cell participation to a given cluster was considered statistically significant if the fraction of SCE in 
    that cluster that activated the cell exceeded the 95th percentile of reshuffled data. 
    If a cell was significantly active in more than one SCE cluster, 
    it was associated to the one in which it participated the most (percentage wise).
    """
    # Count number of participation to each cluster
    # CellP means cell positives, and R means ratio (CellR gives for each cell the
    # percentage of sce during which it's active inside a cluster of SCE.
    cas = cell_assemblies_struct
    if debug_mode:
        for n_id, neuron_label in enumerate(cas.neurons_labels):
            print(f"{n_id} -> {neuron_label}")
    sce_clusters_id = cas.sce_clusters_id
    sce_clusters_labels = cas.sce_clusters_labels
    cellsinpeak = cas.cellsinpeak
    n_cells = cas.n_cells
    n_sce_clusters = cas.n_clusters
    n_sces = cas.n_sces
    cells_p = np.zeros((n_cells, n_sce_clusters))
    cells_r = np.zeros((n_cells, n_sce_clusters))
    for i, cluster_id in enumerate(sce_clusters_id):
        cells_p[:, i] = np.sum(cellsinpeak[:, np.equal(sce_clusters_labels, cluster_id)], axis=1)
        cells_r[:, i] = cells_p[:, i] / np.sum(np.equal(sce_clusters_labels, cluster_id))

    # if debug_mode:
    #     print(f"cells_p {cells_p}")
    #     for cell_id, cell_p in enumerate(cells_p):
    #         print(f"{cell_id}: cas.neurons_labels {cas.neurons_labels[cell_id]} {cells_p}")
    #     print(f"cells_r {cells_r}")

    # Test for statistical significance
    cells_cl = np.zeros((n_sce_clusters, n_cells), dtype="uint8")  # Binary matrix of cell associated to clusters

    for cell in np.arange(n_cells):
        # print(f"cell {cell} {cas.neurons_labels[cell]}:")
        # Random distribution among Clusters
        r_clr = np.zeros((n_sce_clusters, n_surrogate))
        # number of random values to apply, look at how many times the cell spikes in a SCE
        n_rnd = np.sum(cellsinpeak[cell, :])

        for surrogate in np.arange(n_surrogate):
            rand_perm = np.random.permutation(n_sces)
            rand_perm = rand_perm[:n_rnd]
            racer = np.zeros(n_sces, dtype="uint8")
            racer[rand_perm] = 1
            for i, cluster_id in enumerate(sce_clusters_id):
                r_clr[i, surrogate] = np.sum(racer[np.equal(sce_clusters_labels, cluster_id)])
        # sorting r_clr, for each cluster, will sort by ascending number of spikes in a surrogate SCE
        r_clr = np.sort(r_clr, axis=1)
        # / n_sce_clusters is to account for the inter- dependence of the comparisons (Bonferroni correction)
        th_max = r_clr[:, int(n_surrogate * (1 - (0.05 / n_sce_clusters)))]
        # if debug_mode:
            # print(f"int(n_surrogate * (1 - 0.05 / n_sce_clusters)) {int(n_surrogate * (1 - 0.05 / n_sce_clusters))}")
            # print(f"th_max {th_max}")
            # print(f"cells_p[cell, :] {cells_p[cell, :]}")
        for i, cluster_id in enumerate(sce_clusters_id):
            cells_cl[i, cell] = int(cells_p[cell, i] > th_max[i])

    cells_with_no_sce_cluster = np.where(np.sum(cells_cl, axis=0) == 0)[0]  # Cells not in any cluster
    cells_with_one_sce_cluster = np.where(np.sum(cells_cl, axis=0) == 1)[0]
    cells_with_several_sce_cluster = np.where(np.sum(cells_cl, axis=0) >= 2)[0]

    # print(f"cells_cl {cells_cl}")
    if debug_mode:
        print(f"cells_with_no_sce_cluster {cells_with_no_sce_cluster}")
        print(f"cells_with_one_sce_cluster {cells_with_one_sce_cluster}")
        print(f"cells_with_several_sce_cluster {cells_with_several_sce_cluster}")

    # Keep cluster where they participate the most
    for cell_id in cells_with_several_sce_cluster:
        # If a cell was significantly active in more than one SCE cluster,
        # it was associated to the one in which it participated the most (percentage wise)
        index_cl = np.argmax(cells_r[cell_id, :])
        cells_cl[:, cell_id] = 0
        cells_cl[index_cl, cell_id] = 1

    if debug_mode:
        print(f"cells_cl {cells_cl}")

    """
    The overlap between assemblies was quantified by calculating the silhouette value of each cell 
    (with the normalized hamming distance between each cell pair as a dissimilarity metric). 
    A cell was significantly involved in a single assembly if its silhouette value was higher 
    than expected by chance (95th percentile after reshuffling). SCE were finally sorted with respect 
    to their projection onto cell assemblies. A SCE was activating a given neuronal assembly 
    if the number of cells recruited in that assembly was higher than expected by chance (95th percentile after reshuffling)
    """

    # list of list of int representing cell indices that
    # belong to this cluster
    # cell_assemblies_clusters represents the cell assemblies clusters
    # cell_assemblies_clusters is c_0 in Arnaud's code
    cell_assemblies_clusters = list()
    # dict used to map the cell assembly cluster index to the sce cluster index it was formed by
    map_index_cluster_to_original_cluster_id = dict()
    for i, cluster_id in enumerate(sce_clusters_id):
        min_cells_nb = 2
        cells_in_cluster = np.where(cells_cl[i, :])[0]
        # keep cells index for clusters with at least min_cells_nb
        if len(cells_in_cluster) >= min_cells_nb:
            map_index_cluster_to_original_cluster_id[len(cell_assemblies_clusters)] = cluster_id
            cell_assemblies_clusters.append(cells_in_cluster)

    # print(f"c_0 {cell_assemblies_clusters}")

    # Participation rate to its own cluster
    # not used
    cells_r_cl = np.max(cells_r[np.concatenate((cells_with_one_sce_cluster, cells_with_several_sce_cluster)), :],
                        axis=1)

    # Assign sce to groups of cells

    # n_assemblies_cl (NCl) is the number of significant clusters with more than x cells
    # C0 is a dict, key represent cluster number, and value is an array
    # of cells index belonging to the cluster
    n_assemblies_cl = len(cell_assemblies_clusters)
    if debug_mode:
        print(f"n_assemblies_cl {n_assemblies_cl}")
    # Cell count in each cluster
    r_cl = np.zeros((n_assemblies_cl, n_sces))
    p_cl = np.zeros((n_assemblies_cl, n_sces), dtype="uint8")  # binary matrix of sce associated to cell assembly clusters

    for i, cells in enumerate(cell_assemblies_clusters):
        r_cl[i, :] = np.sum(cellsinpeak[cells, :], axis=0)

    # print(f"r_cl {r_cl}")

    r_cln = np.zeros((n_assemblies_cl, n_sces))

    for sce_id in np.arange(n_sces):
        # Random distribution among Clusters
        r_clr = np.zeros((n_assemblies_cl, n_surrogate))
        n_rnd = np.sum(cellsinpeak[:, sce_id])

        for surrogate_id in np.arange(n_surrogate):
            rand_perm = np.random.permutation(n_cells)
            rand_perm = rand_perm[:n_rnd]
            racer = np.zeros(n_cells, dtype="uint8")
            racer[rand_perm] = 1
            for cell_assembly_id, cells in enumerate(cell_assemblies_clusters):
                r_clr[cell_assembly_id, surrogate_id] = np.sum(racer[cells])

        # sorting r_clr, for each cluster, will sort by ascending number of spikes in a surrogate SCE
        r_clr = np.sort(r_clr, axis=1)
        # / n_sce_clusters is for the bonferroni correction
        th_max = r_clr[:, int(n_surrogate * (1 - (0.05 / n_sce_clusters)))]
        # print(f"int(n_surrogate * (1 - 0.05 / n_sce_clusters)) {int(n_surrogate * (1 - 0.05 / n_sce_clusters))}")
        # print(f"th_max {th_max}")
        # print(f"cells_p[j, :] {cells_p[j, :]}")
        for cell_assembly_id, cells in enumerate(cell_assemblies_clusters):
            p_cl[cell_assembly_id, sce_id] = int(r_cl[cell_assembly_id, sce_id] > th_max[cell_assembly_id])
        # Normalize (probability)
        r_cln[:, sce_id] = r_cl[:, sce_id] / np.sum(cellsinpeak[:, sce_id])
    if debug_mode:
        print(f"np.sum(p_cl, axis=1) {np.sum(p_cl, axis=1)}")

    ############################################
    # #### variables for later display purposes
    ############################################
    # give the original cell index
    cells_indices = np.zeros(n_cells, dtype="uint16")
    # give the original sce index
    sce_indices = np.zeros(n_sces, dtype="uint16")
    # give the number of sce in the no-assembly-sce, single-assembly and multiple-assembly groups respectively
    n_sce_in_assembly = np.zeros(3, dtype="uint16")
    # contains the nb of cells in each cell assemblie cluster
    n_cells_in_cell_assemblies_clusters = []
    # contains the nb of sces in sce single cell assembly cluster
    n_cells_in_single_cell_assembly_sce_cl = []
    # contains the nb of sces in sce multiple cell assembly cluster
    n_cells_in_multiple_cell_assembly_sce_cl = []
    # key will be the cell assembly index (such as displayed) and the value is a list of list of 2 elements (first and
    # last sce index(non included) of each sce cluster part of this assembly)
    sces_in_cell_assemblies_clusters = dict()
    # key is a int representing a sce, and value a list representing the id of cell assemblies cluster
    cell_assemblies_cluster_of_multiple_ca_sce = dict()

    cellsinpeak_just_cells_ordered = np.zeros((n_cells, n_sces), dtype="int16")

    # #### ordering cells first   #####
    all_cells = np.arange(n_cells)
    cells_ordered = np.zeros(0, dtype="int16")
    start = 0
    for cells in cell_assemblies_clusters:
        cellsinpeak_just_cells_ordered[start:(start + len(cells)), :] = cellsinpeak[cells, :]
        cells_indices[start:(start + len(cells))] = cells
        n_cells_in_cell_assemblies_clusters.append(len(cells))
        start += len(cells)
        cells_ordered = np.concatenate((cells_ordered, cells))
    cells_left = np.setdiff1d(all_cells, cells_ordered)
    cellsinpeak_just_cells_ordered[start:, :] = cellsinpeak[cells_left, :]
    cells_indices[start:] = cells_left

    cellsinpeak_ordered = np.zeros((n_cells, n_sces), dtype="int16")

    # now ordering sce
    # p_cl = np.zeros((n_cl, n_sces))
    # first we look at sces associated to no cell assembly
    start = 0
    sce_still_to_organized = np.arange(n_sces)
    no_assembly_sce = np.where(np.sum(p_cl, axis=0) == 0)[0]
    if debug_mode:
        print(f"len(no_assembly_sce) {len(no_assembly_sce)}, "
              f"no_assembly_sce {no_assembly_sce}")
    # toto error
    cellsinpeak_ordered[:, start:(start + len(no_assembly_sce))] = cellsinpeak_just_cells_ordered[:, no_assembly_sce]
    sce_indices[start:(start + len(no_assembly_sce))] = no_assembly_sce
    start += len(no_assembly_sce)
    n_sce_in_assembly[0] = len(no_assembly_sce)
    sce_still_to_organized = np.setdiff1d(sce_still_to_organized, no_assembly_sce)


    # then those sces associated to one cell assembly
    # and we organize sces among them as sce clusters in which the cell assemblie belongs
    single_assembly_sce = np.where(np.sum(p_cl, axis=0) == 1)[0]

    if debug_mode:
        print(f"len(single_assembly_sce) {len(single_assembly_sce)}, "
              f"single_assembly_sce {single_assembly_sce}")
    if len(single_assembly_sce) > 0:
        # putting them in the same order as cell assemblies
        sce_already_organized = np.zeros(0, dtype="uint16")
        for cell_assembly_cluster_id in np.arange(len(cell_assemblies_clusters)):
            # sce index that are part of this cluster
            sces_in_cluster = np.where(p_cl[cell_assembly_cluster_id, :])[0]
            # now we check which ones are indeed in the single_assembly_sce group
            sces_in_cluster = (np.intersect1d(sces_in_cluster, single_assembly_sce)).astype(int)
            # print(f"np.intersect1d(sces_in_cluster, sce_already_organized) "
            #       f"{np.intersect1d(sces_in_cluster, sce_already_organized)}")
            if len(sces_in_cluster) > 0:
                cellsinpeak_ordered[:, start:(start + len(sces_in_cluster))] = \
                    cellsinpeak_just_cells_ordered[:, sces_in_cluster]
                sce_indices[start:(start + len(sces_in_cluster))] = sces_in_cluster
                if cell_assembly_cluster_id not in sces_in_cell_assemblies_clusters:
                    sces_in_cell_assemblies_clusters[cell_assembly_cluster_id] = []
                sces_in_cell_assemblies_clusters[cell_assembly_cluster_id].\
                    append((start, (start + len(sces_in_cluster))))

                start += len(sces_in_cluster)
                n_sce_in_assembly[1] += len(sces_in_cluster)
                n_cells_in_single_cell_assembly_sce_cl.append(len(sces_in_cluster))
                sce_already_organized = np.concatenate((sce_already_organized, sces_in_cluster))
        # print(f"np.setdiff1d(single_assembly_sce, sce_already_organized : "
        #       f"{np.setdiff1d(single_assembly_sce, sce_already_organized)}")
        sce_still_to_organized = np.setdiff1d(sce_still_to_organized, sce_already_organized)

    # then those sces associated to multiple cell assembly
    # and we organize sces among them as sce clusters in which the cell assemblie belongs
    multiple_assembly_sce = np.where(np.sum(p_cl, axis=0) > 1)[0]
    if debug_mode:
        print(f"len(multiple_assembly_sce) {len(multiple_assembly_sce)}, "
              f"multiple_assembly_sce {multiple_assembly_sce}")
    if len(multiple_assembly_sce) > 0:
        for sce_id in multiple_assembly_sce:
            cell_assemblies_cluster_of_multiple_ca_sce[sce_id] = np.where(p_cl[:, sce_id])[0]
        sce_already_organized = np.zeros(0, dtype="uint16")
        for cluster_id in np.arange(len(cell_assemblies_clusters)):
            # sce index that are part of this cluster
            sces_in_cluster = np.where(p_cl[cluster_id, :])[0]
            # now we check which ones are indeed in the multiple_assembly_sce group
            sces_in_cluster = (np.intersect1d(sces_in_cluster, multiple_assembly_sce)).astype(int)
            # because they are in multiple cell assemblies, we remove them if they have already been added to a cell
            # assembly cluster for ordering
            sces_in_cluster = (np.setdiff1d(sces_in_cluster, sce_already_organized)).astype(int)
            # print(f"np.intersect1d(sces_in_cluster, sce_already_organized) "
            #       f"{np.intersect1d(sces_in_cluster, sce_already_organized)}")
            if len(sces_in_cluster) > 0:
                cellsinpeak_ordered[:, start:(start + len(sces_in_cluster))] = \
                    cellsinpeak_just_cells_ordered[:, sces_in_cluster]
                sce_indices[start:(start + len(sces_in_cluster))] = sces_in_cluster
                # if cluster_id not in sces_in_cell_assemblies_clusters:
                #     sces_in_cell_assemblies_clusters[cluster_id] = []
                # sces_in_cell_assemblies_clusters[cluster_id].append((start, (start + len(sces_in_cluster))))
                start += len(sces_in_cluster)
                n_sce_in_assembly[2] += len(sces_in_cluster)
                n_cells_in_multiple_cell_assembly_sce_cl.append(len(sces_in_cluster))
                sce_already_organized = np.concatenate((sce_already_organized, sces_in_cluster))
        # print(f"np.setdiff1d(multiple_assembly_sce, sce_already_organized : "
        #       f"{np.setdiff1d(multiple_assembly_sce, sce_already_organized)}")
        sce_still_to_organized = np.setdiff1d(sce_still_to_organized, sce_already_organized)

    if len(sce_still_to_organized) > 0:
        print(f"Something's wrong, sce_still_to_organized {sce_still_to_organized}")

    cas.cellsinpeak_ordered = cellsinpeak_ordered
    cas.n_cells_in_cell_assemblies_clusters = n_cells_in_cell_assemblies_clusters
    cas.n_cells_not_in_cell_assemblies = cas.n_cells - np.sum(n_cells_in_cell_assemblies_clusters)
    cas.cells_indices = cells_indices
    cas.sce_indices = sce_indices
    cas.sces_in_cell_assemblies_clusters = sces_in_cell_assemblies_clusters
    cas.n_sce_in_assembly = n_sce_in_assembly
    cas.n_cells_in_single_cell_assembly_sce_cl = n_cells_in_single_cell_assembly_sce_cl
    cas.n_cells_in_multiple_cell_assembly_sce_cl = n_cells_in_multiple_cell_assembly_sce_cl
    cas.cell_assemblies_cluster_of_multiple_ca_sce = cell_assemblies_cluster_of_multiple_ca_sce


def compute_kmean(neurons_labels, cellsinpeak, n_surrogate, range_n_clusters, param,
                  sliding_window_duration, SCE_times, data_id,
                  fct_to_keep_best_silhouettes=np.mean, keep_only_the_best=True,
                  debug_mode=False):
    best_kmeans_by_cluster, m_cov_sces, surrogate_percentile, cluster_with_best_silhouette_score = \
        clusters_on_sce_from_covnorm(cells_in_sce=cellsinpeak,
                                     n_surrogate=n_surrogate,
                                     fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                                     range_n_clusters=range_n_clusters,
                                     neurons_labels=neurons_labels,
                                     debug_mode=debug_mode)

    range_n_clusters_original = range_n_clusters
    if keep_only_the_best:
        range_n_clusters = [cluster_with_best_silhouette_score]

    # give the indices of the sce cluster for whom the silhouette is > to the 95th percentile from n_surrogate
    # surrogates
    significant_sce_clusters = give_significant_sce_clusters(kmeans_dict=best_kmeans_by_cluster,
                                                             range_n_clusters=range_n_clusters,
                                                             m_sces=m_cov_sces,
                                                             surrogate_percentile=surrogate_percentile)

    # updating range_n_clusters
    range_n_clusters_tmp = []
    for n_cluster in range_n_clusters:
        if n_cluster in significant_sce_clusters:
            range_n_clusters_tmp.append(n_cluster)
    range_n_clusters = np.array(range_n_clusters_tmp)

    cell_assemblies_struct_dict = dict()
    for n_cluster in range_n_clusters:
        kmeans = best_kmeans_by_cluster[n_cluster]
        cas = CellAssembliesStruct(data_id=data_id, sce_clusters_labels=kmeans.labels_, cellsinpeak=cellsinpeak,
                                   sce_clusters_id=significant_sce_clusters[n_cluster], n_clusters=n_cluster,
                                   cluster_with_best_silhouette_score=cluster_with_best_silhouette_score,
                                   param=param, neurons_labels=neurons_labels,
                                   sliding_window_duration=sliding_window_duration,
                                   n_surrogate_k_mean=n_surrogate, SCE_times=SCE_times)
        statistical_cell_assemblies_def(cell_assemblies_struct=cas, debug_mode=debug_mode)
        cell_assemblies_struct_dict[n_cluster] = cas

    # will contains as value list of int (corresponding to sce cluster label, -1 if no sce cluster), the len of
    # the list is n_cells
    # So far, a cell belongs to a cluster if the cluster is significant, the cells spikes at least twice in it,
    # and among the significant cluster, the cell has the highest ratio of spikes in that one (for equal ratios, the
    # highest number of cells determine to which cluster it belongs
    cluster_labels_for_neurons = dict()
    for n_cluster in range_n_clusters:
        cluster_labels = best_kmeans_by_cluster[n_cluster].labels_
        cluster_labels_for_neurons[n_cluster] = \
            find_cluster_labels_for_neurons(cells_in_peak=cellsinpeak,
                                            cluster_labels=cluster_labels, m_sces=m_cov_sces,
                                            significant_clusters=significant_sce_clusters[n_cluster])

    return range_n_clusters, best_kmeans_by_cluster, m_cov_sces, cluster_labels_for_neurons, surrogate_percentile, \
           significant_sce_clusters, cluster_with_best_silhouette_score, cell_assemblies_struct_dict


def compute_and_plot_clusters_raster_kmean_version(labels, activity_threshold, range_n_clusters_k_mean,
                                                   n_surrogate_k_mean,
                                                   spike_nums_to_use, cellsinpeak, data_descr,
                                                   param,
                                                   sliding_window_duration, sce_times_numbers,
                                                   SCE_times, perc_threshold,
                                                   n_surrogate_activity_threshold,
                                                   with_shuffling=True,
                                                   sce_times_bool=None,
                                                   debug_mode=False, keep_only_the_best=True,
                                                   with_cells_in_cluster_seq_sorted=False,
                                                   fct_to_keep_best_silhouettes=np.mean):
    # perc_threshold is the number of percentile choosen to determine the threshold
    #
    # -------- clustering params ------ -----
    # range_n_clusters_k_mean = np.arange(2, 17)
    # n_surrogate_k_mean = 100

    results = compute_kmean(data_id=data_descr, neurons_labels=labels, cellsinpeak=cellsinpeak,
                            n_surrogate=n_surrogate_k_mean, param=param,
                            range_n_clusters=range_n_clusters_k_mean,
                            fct_to_keep_best_silhouettes=fct_to_keep_best_silhouettes,
                            debug_mode=debug_mode, keep_only_the_best=keep_only_the_best,
                            sliding_window_duration=sliding_window_duration,
                            SCE_times=SCE_times)

    if results is None:
        return

    range_n_clusters_k_mean, best_kmeans_by_cluster, m_cov_sces, \
    cluster_labels_for_neurons, surrogate_percentiles, significant_sce_clusters, \
    cluster_with_best_silhouette_score, cas_dict = results


    # printing values
    # for n_cluster in range_n_clusters_k_mean:
    #     kmeans = best_kmeans_by_cluster[n_cluster]
    #     surrogate_percentile = surrogate_percentiles[n_cluster]
    #     cluster_labels = kmeans.labels_
    #     silhouette_avg = metrics.silhouette_score(m_cov_sces, cluster_labels, metric='euclidean')
    #     print(f"{data_descr}, {n_cluster} clusters, avg silh: {silhouette_avg}, surr: {surrogate_percentile}")

    if keep_only_the_best:
        print(f"best number of clusters: {cluster_with_best_silhouette_score}")
        range_n_clusters_k_mean = [cluster_with_best_silhouette_score]

    data_descr_backup = data_descr
    for n_cluster in range_n_clusters_k_mean:
        data_descr = data_descr_backup
        if n_cluster == cluster_with_best_silhouette_score:
            data_descr += "_best_cluster"
        cas = cas_dict[n_cluster]
        cas.neurons_labels = labels
        cas.sliding_window_duration = sliding_window_duration

        cas.plot_cell_assemblies(data_descr=data_descr, spike_nums=spike_nums_to_use,
                                 SCE_times=SCE_times, activity_threshold=activity_threshold,
                                 with_cells_in_cluster_seq_sorted=False,
                                 sce_times_bool=sce_times_bool,
                                 save_formats=["pdf", "png"])

        cas.save_data_on_file(n_clusters=n_cluster)

        # if more than 500 cells, we display only the cells in the cell assemblie for the raster part
        if len(spike_nums_to_use) > 500:
            cas = cas_dict[n_cluster]
            cas.neurons_labels = labels
            cas.sliding_window_duration = sliding_window_duration

            cas.plot_cell_assemblies(data_descr=data_descr, spike_nums=spike_nums_to_use,
                                     SCE_times=SCE_times, activity_threshold=activity_threshold,
                                     with_cells_in_cluster_seq_sorted=False,
                                     sce_times_bool=sce_times_bool,
                                     display_only_cell_assemblies_on_raster=True,
                                     save_formats=["pdf", "png"])

            cas.save_data_on_file(n_clusters=n_cluster)

        # TODO: see while plotting twice cell_assemblies make it bug
        if with_cells_in_cluster_seq_sorted:
            cas.plot_cell_assemblies(data_descr=data_descr + "_seq", spike_nums=spike_nums_to_use,
                                     SCE_times=SCE_times, activity_threshold=activity_threshold,
                                     with_cells_in_cluster_seq_sorted=True,
                                     sce_times_bool=sce_times_bool)

        # this section will order the spike_nums for display purpose
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
            if nb_k == 0:
                continue
            cells_indices = np.where(e)[0]
            if with_cells_in_cluster_seq_sorted and (len(cells_indices) > 2):
                result_ordering = order_spike_nums_by_seq(spike_nums_to_use[cells_indices, :], param,
                                                          debug_mode=debug_mode,
                                                          sce_times_bool=sce_times_bool)
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

        # figure with k-mean results
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
        spike_shape_size = 1
        if len(cell_labels) > 150:
            spike_shape_size = 0.5
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
                           raster_face_color='black',
                           cell_spikes_color='white',
                           horizontal_lines=np.array(cluster_horizontal_thresholds) - 0.5,
                           horizontal_lines_colors=['white'] * len(cluster_horizontal_thresholds),
                           horizontal_lines_sytle="dashed",
                           horizontal_lines_linewidth=[1] * len(cluster_horizontal_thresholds),
                           # vertical_lines=SCE_times,
                           # vertical_lines_colors=['white'] * len(SCE_times),
                           # vertical_lines_sytle="solid",
                           # vertical_lines_linewidth=[0.2] * len(SCE_times),
                           span_area_coords=[SCE_times],
                           span_area_colors=['white'],
                           cells_to_highlight=cells_to_highlight,
                           cells_to_highlight_colors=cells_to_highlight_colors,
                           sliding_window_duration=sliding_window_duration,
                           show_sum_spikes_as_percentage=True,
                           spike_shape="o",
                           spike_shape_size=spike_shape_size,
                           save_formats="pdf",
                           axes_list=[ax1, ax2],
                           SCE_times=SCE_times)

        show_co_var_first_matrix(cells_in_peak=np.copy(cellsinpeak), m_sces=m_cov_sces,
                                 significant_sce_clusters=significant_sce_clusters[n_cluster],
                                 n_clusters=n_cluster, kmeans=best_kmeans_by_cluster[n_cluster],
                                 cluster_labels_for_neurons=cluster_labels_for_neurons[n_cluster],
                                 data_str=data_descr, path_results=param.path_results,
                                 show_silhouettes=True, neurons_labels=labels,
                                 surrogate_silhouette_avg=surrogate_percentiles[n_cluster],
                                 axes_list=[ax5, ax3, ax4], fig_to_use=fig, save_formats=["pdf", "png"])
        plt.close()

        # ######### Plot that show cluster activation

        # do_cluster_activations_computing(cas)

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


def do_cluster_activations_computing(cas):
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
                             data_str=f"{n_cluster}_clusters_raster_participation_{data_descr}",
                             axes_list=[ax2],
                             fig_to_use=fig)

    plot_hist_clusters_by_sce(cluster_particpation_to_sce,
                              data_str=f"{n_cluster}_clusters_hist_percentage_of_network_events",
                              param=param)

    plt.close()
