import numpy as np
import scipy.ndimage.morphology as morphology
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import patches


class CoordClass:
    def __init__(self, coord, nb_lines, nb_col):
        # contour coord
        self.coord = coord
        self.nb_lines = nb_lines
        self.nb_col = nb_col
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

        for i, c in enumerate(self.coord):

            # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
            c = c - 1

            if c.shape[0] == 0:
                print(f'Error: {i} c.shape {c.shape}')
                continue
            c_filtered = c.astype(int)
            bw = np.zeros((self.nb_lines, self.nb_col), dtype="int8")
            # morphology.binary_fill_holes(input
            bw[c_filtered[1, :], c_filtered[0, :]] = 1

            c_x, c_y = ndimage.center_of_mass(bw)
            self.center_coord[i] = (c_y, c_x)

    def plot_cells_map(self, param, data_id, title_option="", connections_dict=None,
                       background_color=(0, 0, 0, 1), default_cells_color=(1, 1, 1, 1.0),
                       dont_fill_cells_not_in_groups=False,
                       link_connect_color="white", link_line_width=1,
                       cell_numbers_color="dimgray", show_polygons=False,
                       cells_to_link=None,
                       fill_polygons=True, cells_groups=None, cells_groups_colors=None, cells_to_hide=None,
                       with_cell_numbers=False, save_formats="png"):
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

                coord = self.coord[cell]
                coord = coord - 1
                # c_filtered = c.astype(int)
                n_coord = len(coord[0, :])
                xy = np.zeros((n_coord, 2))
                for n in np.arange(n_coord):
                    xy[n, 0] = coord[0, n]
                    xy[n, 1] = coord[1, n]
                self.cell_contour = patches.Polygon(xy=xy,
                                                    fill=True, linewidth=0, facecolor=cells_groups_colors[group_index],
                                                    edgecolor=cells_groups_colors[group_index],
                                                    zorder=z_order_cells, lw=2)
                ax.add_patch(self.cell_contour)
                if with_cell_numbers:
                    self.plot_text_cell(cell=cell, cell_numbers_color=cell_numbers_color)

        for cell in cells_not_in_groups:
            if cell in cells_to_hide:
                continue
            coord = self.coord[cell]
            coord = coord - 1
            # c_filtered = c.astype(int)
            n_coord = len(coord[0, :])
            xy = np.zeros((n_coord, 2))
            for n in np.arange(n_coord):
                xy[n, 0] = coord[0, n]
                xy[n, 1] = coord[1, n]
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

        ax.set_ylim(0, 200)
        ax.set_xlim(0, 200)
        ylim = ax.get_ylim()
        # invert Y
        # ax.set_ylim(ylim[::-1])

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
        frame = plt.gca()
        frame.axes.get_xaxis().set_visible(False)
        frame.axes.get_yaxis().set_visible(False)
        # ax.xaxis.set_ticks_position('none')
        # ax.yaxis.set_ticks_position('none')
        #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{data_id}_cell_maps_{title_option}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")
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
