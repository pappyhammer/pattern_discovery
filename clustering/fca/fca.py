"""
    : fca.py
    Created: 10/04/17
    Description: own interpretation of functional clustering

"""

import numpy as np
import numpy.random as rnd
import pattern_discovery.tools.sce_detection as sce_detection
import pattern_discovery.tools.trains as trains_module
import matplotlib.cm as cm

class ClusterTree:
    def __init__(self, clusters_lists, n_cells, max_scale_value, non_significant_color = "black",
                 merge_history_list=None, father=None):
        # if clusters_lists is a list, it should contain 2 elements
        self.clusters_lists = clusters_lists
        self.merge_history_list = merge_history_list
        # the tree might have no father
        self.father = father
        self.max_scale_value = max_scale_value
        self.cluster_nb = None

        #  ######################## plot param ########################
        self.far_left_child_pos = None
        self.far_right_child_pos = None
        self.x_pos = None
        self.y_pos = None
        self.max_y_pos = 10
        if self.father is None:
            # use for plotting, give the x pos for a given cell_id (key)
            self.x_pos_for_cell = dict()
            # use for filling the position
            self.free_x_pos = np.ones(n_cells, dtype="bool")
            # for each pos, gives the number of cell, used for setting x_axis
            self.pos_cells = np.zeros(n_cells, dtype="uint16")
            # key is an int representing the cluster number
            # and value is the top tree of representing the clusters
            self.cluster_dict = dict()
            self.cluster_nb_list = []
        else:
            self.x_pos_for_cell = self.father.x_pos_for_cell
            self.free_x_pos = self.father.free_x_pos
            self.pos_cells = self.father.pos_cells
            self.cluster_dict = self.father.cluster_dict
            self.cluster_nb_list = self.father.cluster_nb_list

        #  ##############################################################

        # scale_vlaue will be NOne if the tree has no child
        self.scale_value = None
        self.cell_id = None

        self.color = non_significant_color

            # each tree can have either no child, or 2 childs
        if isinstance(clusters_lists, int):
            self.no_child = True
            self.cell_id = clusters_lists
            x_pos = np.where(self.free_x_pos)[0][0]
            self.free_x_pos[x_pos] = False
            self.x_pos_for_cell[self.cell_id] = x_pos
            self.pos_cells[x_pos] = self.cell_id
            # print(f"self.cell_id {self.cell_id}")
        else:
            if len(clusters_lists) != 2:
                print(f"len(clusters_lists) != 2")
            self.no_child = False
            first_child_clusters = clusters_lists[0]
            second_child_clusters = clusters_lists[1]

            index_hist = self.find_merge_history_index(first_child_clusters, second_child_clusters)
            merge_history = self.merge_history_list[index_hist]
            del self.merge_history_list[index_hist]
            self.scale_value = merge_history[2]
            nb_elements=0

            if isinstance(first_child_clusters, int):
                self.first_child = ClusterTree(clusters_lists=first_child_clusters, max_scale_value=max_scale_value,
                                               father=self, n_cells=n_cells)
            else:
                nb_elements = self.nb_cells_in_list(first_child_clusters)
                self.first_child = ClusterTree(clusters_lists=first_child_clusters,n_cells=n_cells,
                                               max_scale_value=max_scale_value,
                                               merge_history_list=self.merge_history_list,
                                               father=self)

            if isinstance(second_child_clusters, int):
                self.second_child = ClusterTree(clusters_lists=second_child_clusters, father=self,
                                               max_scale_value=max_scale_value,
                                               n_cells=n_cells)
            else:
                # nb_elements = self.nb_cells_in_list(second_child_clusters)
                self.second_child = ClusterTree(clusters_lists=second_child_clusters, n_cells=n_cells,
                                                max_scale_value=max_scale_value,
                                                merge_history_list=self.merge_history_list,
                                                father=self)
        # things that need to be done after all childs have been created
        if self.father is None:
            nb_intersections = self.get_nb_intersections()
            self.significant_threshold = self.set_y_pos(nb_intersections=nb_intersections)
            self.compute_clusters()

            n_clusters = len(self.cluster_nb_list)

            self.set_colors(n_clusters=n_clusters)


    def set_colors(self, n_clusters):
        if self.cluster_nb is not None:
            self.color = cm.nipy_spectral(float(self.cluster_nb + 1) / n_clusters)
        if not self.no_child:
            self.first_child.set_colors(n_clusters)
            self.second_child.set_colors(n_clusters)

    def compute_clusters(self):
        if self.father is not None:
            if self.father.cluster_nb is not None:
                self.cluster_nb = self.father.cluster_nb

        if self.no_child:
            return

        # determining cluster
        if (self.cluster_nb is None) and (self.scale_value >= 1) and (self.are_child_significant()):
            self.cluster_nb = len(self.cluster_nb_list)
            self.cluster_nb_list.append(self.cluster_nb)
            self.cluster_dict[self.cluster_nb] = self

        self.first_child.compute_clusters()
        self.second_child.compute_clusters()



    def get_cells_id(self):
        """

        :return: a list of int representing the cells index
        """
        cells = []
        if self.no_child:
            return [self.cell_id]
        cells.extend(self.first_child.get_cells_id())
        cells.extend(self.second_child.get_cells_id())

        return cells

    def set_y_pos(self, nb_intersections):
        # using breadth first search technique
        actual_y_pos = self.max_y_pos
        # to determine the position of last intersection being significant, we look
        # first index is the y_pos, second one is the scale_value
        pos_significant_threshold = [100, 100]

        current_depth = 0
        nodes_to_expend = {current_depth: [self]}
        nodes_at_depth = dict()
        nodes_at_depth[current_depth] = []
        nodes_at_depth[current_depth].append(self)
        while True:
            current_node = nodes_to_expend[current_depth][0]
            nodes_to_expend[current_depth] = nodes_to_expend[current_depth][1:]
            if current_node.no_child:
                current_node.y_pos = 0
            else:
                current_node.y_pos = actual_y_pos
                actual_y_pos -= (self.max_y_pos/nb_intersections)
                if (current_node.scale_value >= 1) and (current_node.are_child_significant()):
                    if current_node.scale_value < pos_significant_threshold[1]:
                        pos_significant_threshold[1] = current_node.scale_value
                        pos_significant_threshold[0] = current_node.y_pos
                    if current_node.scale_value == pos_significant_threshold[1]:
                        if current_node.y_pos > pos_significant_threshold[0]:
                            pos_significant_threshold[0] = current_node.y_pos

            if not current_node.no_child:
                if (current_depth + 1) not in nodes_at_depth:
                    nodes_at_depth[current_depth + 1] = []
                nodes_at_depth[current_depth + 1].append(current_node.first_child)
                nodes_at_depth[current_depth + 1].append(current_node.second_child)

            # we need to check how many leafs are in the tree at this current_depth
            # and remove the one with the lower score
            # adding to nodes_to_expend only the best nodes

            if len(nodes_to_expend[current_depth]) == 0:
                current_depth += 1
                if current_depth in nodes_at_depth:
                    nodes_to_expend[current_depth] = nodes_at_depth[current_depth]
                else:
                    break
        # return significant threshold position
        return pos_significant_threshold[0] + ((self.max_y_pos/nb_intersections) / 2)

    def get_nb_intersections(self):
        if self.no_child:
            return 0
        return 1 + self.first_child.get_nb_intersections() + self.second_child.get_nb_intersections()

    def are_child_significant(self):
        """
        return true if all child have a scale_value >=1
        :return:
        """
        if self.no_child:
            return True
        if self.scale_value < 1:
            return False
        return (self.first_child.are_child_significant() and self.second_child.are_child_significant())

    def plot_cluster(self, ax, with_scale_value=False):
        if self.no_child:
            return
        else:
            x_pos = self.get_x_pos()

            ax.hlines(self.y_pos, self.first_child.get_x_pos(), self.second_child.get_x_pos(),
                      color=self.color,
                      linewidth=4)
            if with_scale_value:
                ax.text(x_pos, self.y_pos-0.2, f'{np.round(self.scale_value, 2)}', horizontalalignment='center',
                        verticalalignment = 'center')

            y_bottom = self.first_child.y_pos
            ax.vlines(self.first_child.get_x_pos(), y_bottom, self.y_pos,
                      color=self.color,
                      linewidth=4)

            y_bottom = self.second_child.y_pos
            ax.vlines(self.second_child.get_x_pos(), y_bottom, self.y_pos,
                      color=self.color,
                      linewidth=4)

            self.first_child.plot_cluster(ax, with_scale_value=with_scale_value)
            self.second_child.plot_cluster(ax, with_scale_value=with_scale_value)

    def get_x_pos(self):
        if self.x_pos is None:
            self.far_left_child_pos = self.get_first_x_pos()
            self.far_right_child_pos = self.get_last_x_pos()
            self.x_pos = np.mean((self.far_left_child_pos, self.far_right_child_pos))
        return self.x_pos


    def get_first_x_pos(self):
        """
        Look for the pos of the child the most on the left of the dendogram from this tree
        :return:
        """
        if self.no_child:
            self.x_pos = self.x_pos_for_cell[self.cell_id]
            return self.x_pos
        else:
            self.far_left_child_pos = self.first_child.get_first_x_pos()
            return self.far_left_child_pos

    def get_last_x_pos(self):
        """
        Look for the pos of the child the most on the left of the dendogram from this tree
        :return:
        """
        if self.no_child:
            self.x_pos = self.x_pos_for_cell[self.cell_id]
            return self.x_pos
        else:
            self.far_right_child_pos = self.second_child.get_last_x_pos()
            return self.far_right_child_pos

    def find_merge_history_index(self, first_child_clusters, second_child_clusters):
        for index, merge_history in enumerate(self.merge_history_list):
            if (merge_history[0] == first_child_clusters) and (merge_history[1] == second_child_clusters):
                return index
        return None


    def nb_cells_in_list(self, cells):
        if isinstance(cells, list):
            if len(cells) == 0:
                return 1
            next_sum = 0
            for item in cells:
                next_sum += self.nb_cells_in_list(item)
            return 1 + next_sum
        else:
            return 0


def get_min_max_scale_from_merge_history(merge_history):
    min_scale = 100
    max_scale = 0

    for merge_data in merge_history:
        scale_value = merge_data[2]
        min_scale = np.min((scale_value, min_scale))
        max_scale = np.max((scale_value, max_scale))

    return min_scale, max_scale

def average_minimum_distance(train1, train2):
    """
        Compute the average minimum distance between spike trains train 1 and train 2
        trains should be in the form train[i] = T ith spike occurs at time T
    :param train1: first train to be compared
    :param train2: second train
    :return: distance positive float

    """
    dist1 = 0
    for spike_t in train1:
        diff_spike = train2 - spike_t
        delta = np.min(np.abs(diff_spike))
        dist1 += delta
    dist2 = 0
    for spike_t in train2:
        diff_spike = train1 - spike_t
        delta = np.min(np.abs(diff_spike))
        dist2 += delta

    return 0.5 * (dist1 + dist2)


def jitter_spike_train(train, sigma):
    """
        Compute new spike trains jitterd by gaussian (centered) noise with sd  sigma
        train should be in the form train[i] = T ith spike occurs at time T
    :param train:
    :param sigma:
    :return: new spike trains - with same number of spike len = len(train)

    """

    n = len(train)
    new_train = train.copy()
    new_train += sigma * rnd.randn(n)
    return new_train


def jitter_data_set(train_list, sigma):
    """
        create new train list with jittered spike trains
    :param train_list: the spike train list to be jittered
    :param sigma: noise of jittering
    :return: new jittered train list

    """
    jittered_list = []
    for train in train_list:
        new_train = jitter_spike_train(train, sigma)
        jittered_list.append(new_train)
    return jittered_list


def distance_data_set(train_list):
    """
        Compute distance matrix for each pair of trains
    :param train_list: list of train in the in the form train[i] = T ith spike occurs at time T
    :return: a len(train_list) square matrix with distance
    """
    ntrains = len(train_list)
    distance_matrix = np.zeros((ntrains, ntrains))
    for i in range(ntrains):
        for j in range(i+1, ntrains):
            train1 = train_list[i]
            train2 = train_list[j]
            d = average_minimum_distance(train1, train2)
            distance_matrix[i, j] = d
            distance_matrix[j, i] = d
    return distance_matrix


def update_distance_data_set(distance_matrix, train_list, idx_train1, surrogate_train):
    """
        update the distance matrix for a pair a train assuming the train list contains new (with the merged train)
        at position idx_train1 - we remove row and column idx_train2
    :param train_list: list of train in the in the form train[i] = T ith spike occurs at time T
    :return: a len(train_list) square matrix with distance
    """
    ntrains = len(train_list)
    # print(f"np.shape(distance_matrix) {np.shape(distance_matrix)} ntrains {ntrains}")
    for j in np.arange(ntrains):
        train2 = train_list[j]
        d = average_minimum_distance(surrogate_train, train2)
        distance_matrix[idx_train1, j] = d
        distance_matrix[j, idx_train1] = d
    return distance_matrix


def scaled_value(cdf, v, ci=0.05):
    """
        From cdf value compute the distance to the 95% interval (low so 5%)
    :param cdf: the sorted vector of values
    :param v: the value to find the significance level
    :param ci: confidence interval - default 5%
    :return: a float os scaled value
    """
    n = len(cdf)
    n2 = n/2.0
    n5 = n * ci
    nv = len(cdf[cdf < v])
    return max(0, (n2-nv)/(n2-n5))


def scaled_significance_matrix(train_list, cdf_matrix):

    n_surrogate, nx, ny = cdf_matrix.shape
    distance_matrix = distance_data_set(train_list)
    scaled_matrix = np.zeros((nx, nx))
    for i in range(nx):
        for j in range(i+1, nx):
            scaled_matrix[i, j] = scaled_value(cdf_matrix[:, i, j], distance_matrix[i, j])
    return scaled_matrix


def merge_trains(train1, train2):
    """
        Merge spike train in the form train[i] = T ith spike occurs at time T
    :param train1:
    :param train2:
    :return: a merged spike train in the form train[i] = T ith spike occurs at time T
    """
    x = np.concatenate((train1, train2))
    x.sort()
    return x


def create_surrogate_dataset(train_list, nsurrogate, sigma):
    surrogate_data_set = []
    for i in range(nsurrogate):
        surrogate_data_set.append(jitter_data_set(train_list, sigma))
    return surrogate_data_set


def update_surrogate_dataset(surrogate_data_set, sigma, merged_train, idx_train1, idx_train2):
    nsurrogate = len(surrogate_data_set)
    for i in range(nsurrogate):
        train_list = surrogate_data_set[i]
        train_list.pop(idx_train2)
        train_list[idx_train1] = jitter_spike_train(merged_train, sigma)
    return surrogate_data_set


def cdf_distance(train_list, surrogate_data_set):
    """
        create for each pair of train a cumulative distance function for distance based on the surrogate
        data set
        - first nsurrogate surrogate data sets are created
        - then for each surrogate data set a distance matrix is created
        - for each pair - entry in the matrix - a cdf if computed
    :param train_list:
    :param sigma:
    :param nsurrogate:
    :return:
    """
    ntrains = len(train_list)
    nsurrogate = len(surrogate_data_set)

    cdf_matrix = np.zeros((nsurrogate, ntrains, ntrains))
    for i in range(nsurrogate):
        print(f"surrogate {i}")
        surrogate_distance_matrix = distance_data_set(surrogate_data_set[i])
        cdf_matrix[i, :, :] = surrogate_distance_matrix

    # sort each entry to create the cdf
    for i in range(ntrains):
        for j in range(i+1, ntrains):
            cdf_matrix[:, i, j].sort()
            cdf_matrix[:, j, i].sort()
    return cdf_matrix


def update_cdf_distance(cdf_matrix, surrogate_data_set, train_list, idx_train1, idx_train2):
    """
       update  cumulative distance function for distance based on the surrogate
        data set by removing idx_train2 and usinf idx_train1 as placeholder for merged
    """
    ntrains = len(train_list)
    nsurrogate = cdf_matrix.shape[0]
    cdf_matrix = np.delete(cdf_matrix, idx_train2, 1)
    cdf_matrix = np.delete(cdf_matrix, idx_train2, 2)
    for i in range(nsurrogate):
        surrogate_train = surrogate_data_set[i][idx_train1]
        cdf_matrix[i, :, :] = update_distance_data_set(cdf_matrix[i, :, :], train_list, idx_train1, surrogate_train)
    # sort each entry to create the cdf
    for i in range(ntrains):
        for j in range(i+1, ntrains):
            cdf_matrix[:, i, j].sort()
            cdf_matrix[:, j, i].sort()
    return cdf_matrix


def functional_clustering_algorithm(train_list, nsurrogate, sigma, early_stop=True, rolling_surrogate=False):
    """
        Main clustering algorithm
    :param train_list:
    :param nsurrogate:
    :param sigma:
    :return:
    """
    print("starting clustering")
    done = False
    ntrain = len(train_list)
    current_train_list = train_list[:]
    current_cluster = list(np.arange(ntrain))
    merge_history = []
    nstep = 0
    if rolling_surrogate:
        min_time, max_time = trains_module.get_range_train_list(train_list)
        surrogate_data_set = sce_detection.create_surrogate_dataset(train_list=train_list, nsurrogate=nsurrogate,
                                                      min_value=min_time, max_value=max_time)
    else:
        # original method by Feldt
        surrogate_data_set = create_surrogate_dataset(train_list, nsurrogate, sigma)
    cdf_matrix = cdf_distance(current_train_list, surrogate_data_set)

    while not done:
        print(f"doing step {nstep}")
        print("computing cdf")
        scale_matrix = scaled_significance_matrix(current_train_list, cdf_matrix)
        maximum_scale = np.max(scale_matrix)
        print(f"max scale {maximum_scale}")
        if early_stop and maximum_scale < 1.0:
            print(f"early_stop maximum_scale{maximum_scale}, len(current_train_list){len(current_train_list)}")
            done = True
        else:
            i, j = np.unravel_index(np.argmax(scale_matrix), scale_matrix.shape)

            if i == j:
                # part not included in the original algorithm
                # usually means that scale_matrix contains only 0.00
                i = 0
                j = 1
                # break

            if i > j:
                i, j = j, i

            merge_history.append([current_cluster[i], current_cluster[j], maximum_scale])
            new_train = merge_trains(current_train_list[i], current_train_list[j])
            current_cluster[i] = [current_cluster[i], current_cluster[j]]
            current_cluster.pop(j)
            new_list = []
            current_train = len(current_train_list)
            for ix in range(current_train):
                if ix == i:
                    new_list.append(new_train)
                elif ix == j:
                    pass
                else:
                    new_list.append(current_train_list[ix])
            current_train_list = new_list[:]
            if len(current_train_list) == 1 :
                done = True
            else:
                surrogate_data_set = update_surrogate_dataset(surrogate_data_set, sigma, new_train, i, j)
                cdf_matrix = update_cdf_distance(cdf_matrix,  surrogate_data_set, current_train_list, i, j)
        nstep += 1
    return merge_history, current_cluster


def create_linkage(n_nodes, merge_history):
    """
        Create the Z matrix for dendrogram plotting
    :param merge_history: the merge history as list of lists
    :return: a (n-1) * 4 matrix as coded in sklearn.clustering.linkage function
    """
    Z = []
    cluster_list = range(n_nodes)
    cluster_size = [1] * n_nodes

    for merge in merge_history:
        cluster1 = merge[0]
        cluster2 = merge[1]
        if cluster1 not in cluster_list:
            cluster_list.append(cluster1)
            cl1, cl2 = cluster1
            id1 = cluster_size[cluster_list.index(cl1)]
            id2 = cluster_size[cluster_list.index(cl2)]
            cluster1_size = cluster_size[id1] + cluster_size[id2]
            cluster_size.append(cluster1_size)

        if cluster2 not in cluster_list:
            cluster_list.append(cluster2)
            cl1, cl2 = cluster2
            id1 = cluster_size[cluster_list.index(cl1)]
            id2 = cluster_size[cluster_list.index(cl2)]
            cluster2_size = cluster_size[id1] + cluster_size[id2]
            cluster_size.append(cluster2_size)
        print(f"len(cluster_size) {len(cluster_size)}")
        ix1 = cluster_list.index(cluster1)
        ix2 = cluster_list.index(cluster2)
        Z.append([ix1, ix2, 1.0, 1])
    return Z

