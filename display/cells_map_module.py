import numpy as np
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import patches
from shapely import geometry
import PIL
from PIL import ImageDraw
import shapely as shapely
import math


class CoordClass:
    def __init__(self, coord, nb_lines, nb_col, from_suite_2p=False):
        # contour coord
        self.coord = coord
        if nb_lines is None:
            self.nb_lines = 200
        else:
            self.nb_lines = nb_lines
        if nb_col is None:
            self.nb_col = 200
        else:
            self.nb_col = nb_col
        self.n_cells = len(self.coord)
        # dict of tuples, key is the cell #, cell center coord x and y (x and y are inverted for imgshow)
        self.center_coord = dict()
        self.img_filled = None
        self.img_contours = None
        # used in case some cells would be removed, to we can update the centers accordingly
        self.neurons_removed = []
        # compute_center_coord() will be called when a mouseSession will be created
        # self.compute_center_coord()
        self.cells_groups = None
        self.cells_groups_colors = None
        # shapely polygons
        self.cells_polygon = dict()
        # first key is an int representing the number of the cell, and value is a list of cells it interesects
        self.intersect_cells = dict()

        for cell, c in enumerate(self.coord):

            if not from_suite_2p:
                # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
                c = c - 1
            c = c.astype(int)
            self.coord[cell] = c

            if c.shape[0] == 0:
                print(f'Error: {cell} c.shape {c.shape}')
                continue

            c_filtered = c.astype(int)
            bw = np.zeros((self.nb_col, self.nb_lines), dtype="int8")
            # morphology.binary_fill_holes(input
            bw[c_filtered[0, :], c_filtered[1, :]] = 1

            n_coord = len(c_filtered[0, :])
            coord_list_tuple = []
            for n in np.arange(n_coord):
                coord_list_tuple.append((c_filtered[0, n], c_filtered[1, n]))

            # buffer(0) or convex_hull could be used if the coord are a list of points not
            # in the right order. However buffer(0) return a MultiPolygon with no coord available.
            self.cells_polygon[cell] = geometry.Polygon(coord_list_tuple)  # .convex_hull # buffer(0)
            # self.coord[cell] = np.array(self.cells_polygon[cell].exterior.coords).transpose()

            c_x, c_y = ndimage.center_of_mass(bw)
            self.center_coord[cell] = (c_x, c_y)

            # if (cell == 0) or (cell == 159):
            #     print(f"cell {cell} fig")
            #     fig, ax = plt.subplots(nrows=1, ncols=1,
            #                            gridspec_kw={'height_ratios': [1]},
            #                            figsize=(5, 5))
            #     ax.imshow(bw)
            #     plt.show()
            #     plt.close()

        for cell_1 in np.arange(self.n_cells-1):
            if cell_1 not in self.intersect_cells:
                self.intersect_cells[cell_1] = set()
            for cell_2 in np.arange(cell_1+1, self.n_cells):
                if cell_2 not in self.intersect_cells:
                    self.intersect_cells[cell_2] = set()
                poly_1 = self.cells_polygon[cell_1]
                poly_2 = self.cells_polygon[cell_2]
                # if it intersects and not only touches if adding and (not poly_1.touches(poly_2))
                # try:
                if poly_1.intersects(poly_2):
                    self.intersect_cells[cell_2].add(cell_1)
                    self.intersect_cells[cell_1].add(cell_2)
                # except shapely.errors.TopologicalError:
                #     print(f"cell_1 {cell_1}, cell_2 {cell_2}")
                #     print(f"cell_1 {poly_1.is_valid}, cell_2 {poly_2.is_valid}")
                #     poly_1 = poly_1.buffer(0)
                #     poly_2 = poly_2.buffer(0)
                #     print(f"cell_1 {poly_1.is_valid}, cell_2 {poly_2.is_valid}")
                #     raise Exception("shapely.errors.TopologicalError")

        # print(f"n_cells {self.n_cells}")
        # n_intersecting_cells = 0
        # n_dict = dict()
        # for cell in np.arange(self.n_cells):
        #     n = len(self.intersect_cells[cell])
        #     if n > 0:
        #         n_intersecting_cells += 1
        #         n_dict[n] = n_dict.get(n, 0) + 1
        # print(f"n_intersecting_cells {n_intersecting_cells}")
        # print(f"n_dict {n_dict}")

    def get_cell_mask(self, cell, dimensions):
        """

        :param cell:
        :param dimensions: height x width
        :return:
        """
        poly_gon = self.cells_polygon[cell]
        img = PIL.Image.new('1', (dimensions[1], dimensions[0]), 0)
        ImageDraw.Draw(img).polygon(list(poly_gon.exterior.coords), outline=1,
                                    fill=1)
        return np.array(img)

    def match_cells_indices(self, coord_obj, param, plot_title_opt=""):
        """

        :param coord_obj: another instanc of coord_obj
        :return: a 1d array, each index corresponds to the index of a cell of coord_obj, and map it to an index to self
        or -1 if no cell match
        """
        mapping_array = np.zeros(len(coord_obj.coord), dtype='int16')
        for cell in np.arange(len(coord_obj.coord)):
            c_x, c_y = coord_obj.center_coord[cell]
            distances = np.zeros(len(self.coord))
            for self_cell in np.arange(len(self.coord)):
                self_c_x, self_c_y = self.center_coord[self_cell]
                # then we calculte the cartesian distance to all other cells
                distances[self_cell] = math.sqrt((self_c_x - c_x) ** 2 + (self_c_y - c_y) ** 2)
            if np.min(distances) <= 2:
                mapping_array[cell] = np.argmin(distances)
            else:
                mapping_array[cell] = -1
        plot_result = True
        if plot_result:
            fig, ax = plt.subplots(nrows=1, ncols=1,
                                   gridspec_kw={'height_ratios': [1]},
                                   figsize=(20, 20))

            ax.set_facecolor("black")

            # dark blue
            other_twin_color = list((0.003, 0.313, 0.678, 1.0))
            n_twins = 0
            # red
            other_orphan_color = list((1, 0, 0, 1.0))
            n_other_orphans = 0
            # light blue
            self_twin_color = list((0.560, 0.764, 1, 1.0))
            # green
            self_orphan_color = list((0.278, 1, 0.101, 1.0))
            n_self_orphans = 0
            # blue = "cornflowerblue"
            # cmap.set_over('red')
            with_edge = True
            edge_line_width = 1
            z_order_cells = 12
            for cell in np.arange(len(coord_obj.coord)):
                xy = coord_obj.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    edge_color = "white"
                else:
                    edge_color = "white"
                    line_width = 0
                # allow to set alpha of the edge to 1
                if mapping_array[cell] >= 0:
                    # dark blue
                    face_color = other_twin_color
                    n_twins += 1
                else:
                    # red
                    face_color = other_orphan_color
                    n_other_orphans += 1
                face_color[3] = 0.8
                face_color = tuple(face_color)
                cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=line_width,
                                                    facecolor=face_color,
                                                    edgecolor=edge_color,
                                                    zorder=z_order_cells)  # lw=2
                ax.add_patch(cell_contour)
            for cell in np.arange(len(self.coord)):
                xy = self.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    edge_color = "white"
                else:
                    edge_color = "white"
                    line_width = 0
                # allow to set alpha of the edge to 1
                if cell in mapping_array:
                    # light blue
                    face_color = self_twin_color
                else:
                    # green
                    face_color = self_orphan_color
                    n_self_orphans += 1
                face_color[3] = 0.8
                face_color = tuple(face_color)
                cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=line_width,
                                                    facecolor=face_color,
                                                    edgecolor=edge_color,
                                                    zorder=z_order_cells)  # lw=2
                ax.add_patch(cell_contour)
            fontsize = 12
            plt.text(x=190, y=180,
                     s=f"{n_twins}", color=self_twin_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=185,
                     s=f"{n_self_orphans}", color=self_orphan_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=190,
                     s=f"{n_twins}", color=other_twin_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            plt.text(x=190, y=195,
                     s=f"{n_other_orphans}", color=other_orphan_color, zorder=22,
                     ha='center', va="center", fontsize=fontsize, fontweight='bold')
            ax.set_ylim(0, self.nb_lines)
            ax.set_xlim(0, self.nb_col)
            ylim = ax.get_ylim()
            # invert Y
            ax.set_ylim(ylim[::-1])
            plt.setp(ax.spines.values(), color="black")
            frame = plt.gca()
            frame.axes.get_xaxis().set_visible(False)
            frame.axes.get_yaxis().set_visible(False)
            save_format = "png"
            fig.savefig(f'{param.path_results}/cells_map_{plot_title_opt}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")
            # plt.show()
            plt.close()
        return mapping_array

    def plot_cells_map(self, param, data_id, title_option="", connections_dict=None,
                       background_color=(0, 0, 0, 1), default_cells_color=(1, 1, 1, 1.0),
                       default_edge_color="white",
                       dont_fill_cells_not_in_groups=False,
                       link_connect_color="white", link_line_width=1,
                       cell_numbers_color="dimgray", show_polygons=False,
                       cells_to_link=None, edge_line_width=2, cells_alpha=1,
                       fill_polygons=True, cells_groups=None, cells_groups_colors=None,
                       cells_groups_alpha=None,
                       cells_to_hide=None,
                       cells_groups_edge_colors=None, with_edge=False,
                       with_cell_numbers=False, save_formats="png",
                       save_plot=True, return_fig=False):
        """

        :param connections_dict: key is an int representing a cell number, and value is a dict representing the cells it
        connects to. The key is a cell is connected too, and the value represent the strength of the connection (like how
        many
        times it connects to it)
        :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
        :return:
        """

        cells_center = self.center_coord
        n_cells = len(self.coord)
        if cells_to_hide is None:
            cells_to_hide = []

        cells_in_groups = []
        if cells_groups is not None:
            for group_id, cells_group in enumerate(cells_groups):
                cells_in_groups.extend(cells_group)
        cells_in_groups = np.array(cells_in_groups)
        cells_not_in_groups = np.setdiff1d(np.arange(n_cells), cells_in_groups)

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 20))

        ax.set_facecolor(background_color)

        # blue = "cornflowerblue"
        # cmap.set_over('red')
        z_order_cells = 12
        for group_index, cell_group in enumerate(cells_groups):
            for cell in cell_group:
                if cell in cells_to_hide:
                    continue

                xy = self.coord[cell].transpose()
                if with_edge:
                    line_width = edge_line_width
                    if cells_groups_edge_colors is None:
                        edge_color = default_edge_color
                    else:
                        edge_color = cells_groups_edge_colors[group_index]
                else:
                    edge_color = cells_groups_colors[group_index]
                    line_width = 0
                # allow to set alpha of the edge to 1
                face_color = list(cells_groups_colors[group_index])
                # changing alpha
                if cells_groups_alpha is not None:
                    face_color[3] = cells_groups_alpha[group_index]
                else:
                    face_color[3] = cells_alpha
                face_color = tuple(face_color)
                self.cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=line_width,
                                                    facecolor=face_color,
                                                    edgecolor=edge_color,
                                                    zorder=z_order_cells) # lw=2
                ax.add_patch(self.cell_contour)
                if with_cell_numbers:
                    self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color)

        for cell in cells_not_in_groups:
            if cell in cells_to_hide:
                continue
            xy = self.coord[cell].transpose()
            # face_color = default_cells_color
            # if dont_fill_cells_not_in_groups:
            #     face_color = None
            self.cell_contour = patches.Polygon(xy=xy,
                                                fill=not dont_fill_cells_not_in_groups,
                                                linewidth=0, facecolor=default_cells_color,
                                                edgecolor=default_cells_color,
                                                zorder=z_order_cells, lw=2)
            ax.add_patch(self.cell_contour)

            if with_cell_numbers:
                self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color)

        ax.set_ylim(0, self.nb_lines)
        ax.set_xlim(0, self.nb_col)
        ylim = ax.get_ylim()
        # invert Y
        ax.set_ylim(ylim[::-1])

        if (connections_dict is not None) :
            zorder_lines = 15
            for neuron in connections_dict.keys():
                # plot a line to all out of the neuron
                for connected_neuron, nb_connexion in connections_dict[neuron].items():
                    line_width = link_line_width + np.log(nb_connexion)

                    c_x = cells_center[neuron][0]
                    c_y = cells_center[neuron][1]
                    c_x_c = cells_center[connected_neuron][0]
                    c_y_c = cells_center[connected_neuron][1]

                    line = plt.plot((c_x, c_x_c), (c_y, c_y_c), linewidth=line_width, c=link_connect_color,
                                    zorder=zorder_lines)[0]

        if (self.cells_groups is not None) and show_polygons:
            for group_id, cells in enumerate(self.cells_groups):
                points = np.zeros((2, len(cells)))
                for cell_id, cell in enumerate(cells):
                    c_x, c_y = cells_center[cell]
                    points[0, cell_id] = c_x
                    points[1, cell_id] = c_y
                # finding the convex_hull for each group
                xy = convex_hull(points=points)
                # xy = xy.transpose()
                # print(f"xy {xy}")
                # xy is a numpy array with as many line as polygon point
                # and 2 columns: x and y coord of each point
                face_color = list(self.cells_groups_colors[group_id])
                # changing alpha
                face_color[3] = 0.3
                face_color = tuple(face_color)
                # edge alpha will be 1
                poly_gon = patches.Polygon(xy=xy,
                                           fill=fill_polygons, linewidth=0, facecolor=face_color,
                                           edgecolor=self.cells_groups_colors[group_id],
                                           zorder=15, lw=3)
                ax.add_patch(poly_gon)

        # plt.title(f"Cells map {data_id} {title_option}")

        # ax.set_frame_on(False)

        # invert Y
        ax.set_ylim(ylim[::-1])
        plt.setp(ax.spines.values(), color=background_color)
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        # ax.xaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks_position('none')
        #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,
        if save_plot:
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{param.path_results}/{data_id}_cell_maps_{title_option}'
                            f'_{param.time_str}.{save_format}',
                            format=f"{save_format}")
        if return_fig:
            return fig
        else:
            plt.close()

    def plot_text_cell(self, cell, cell_numbers_color):
        fontsize = 6
        if cell >= 100:
            fontsize = 4
        elif cell >= 10:
            fontsize = 5

        c_x_c = self.center_coord[cell][0]
        c_y_c = self.center_coord[cell][1]

        plt.text(x=c_x_c, y=c_y_c,
                 s=f"{cell}", color=cell_numbers_color, zorder=22,
                 ha='center', va="center", fontsize=fontsize + 2, fontweight='bold')


def _angle_to_point(point, centre):
    '''calculate angle in 2-D between points and x axis'''
    delta = point - centre
    res = np.arctan(delta[1] / delta[0])
    if delta[0] < 0:
        res += np.pi
    return res


def area_of_triangle(p1, p2, p3):
    '''calculate area of any triangle given co-ordinates of the corners'''
    return np.linalg.norm(np.cross((p2 - p1), (p3 - p1))) / 2.


def convex_hull(points, smidgen=0.0075):
    '''
    from: https://stackoverflow.com/questions/17553035/draw-a-smooth-polygon-around-data-points-in-a-scatter-plot-in-matplotlib
    Calculate subset of points that make a convex hull around points
    Recursively eliminates points that lie inside two neighbouring points until only convex hull is remaining.

    :Parameters:
    points : ndarray (2 x m)
    array of points for which to find hull
    use pylab to show progress?
    smidgen : float
    offset for graphic number labels - useful values depend on your data range

    :Returns:
    hull_points : ndarray (2 x n)
    convex hull surrounding points
    '''

    n_pts = points.shape[1]
    # assert(n_pts > 5)
    centre = points.mean(1)

    angles = np.apply_along_axis(_angle_to_point, 0, points, centre)
    pts_ord = points[:, angles.argsort()]

    pts = [x[0] for x in zip(pts_ord.transpose())]
    prev_pts = len(pts) + 1
    k = 0
    while prev_pts > n_pts:
        prev_pts = n_pts
        n_pts = len(pts)
        i = -2
        while i < (n_pts - 2):
            Aij = area_of_triangle(centre, pts[i], pts[(i + 1) % n_pts])
            Ajk = area_of_triangle(centre, pts[(i + 1) % n_pts], pts[(i + 2) % n_pts])
            Aik = area_of_triangle(centre, pts[i], pts[(i + 2) % n_pts])
            if Aij + Ajk < Aik:
                del pts[i + 1]
            i += 1
            n_pts = len(pts)
        k += 1
    return np.asarray(pts)
