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
        self.center_coord = list()
        self.img_filled = None
        self.img_contours = None
        # used in case some cells would be removed, to we can update the centers accordingly
        self.neurons_removed = []
        # compute_center_coord() will be called when a mouseSession will be created
        # self.compute_center_coord()
        self.cells_groups = None
        self.cells_groups_colors = None

    def compute_center_coord(self, cells_groups=None, cells_groups_colors=None, cells_to_hide=None):
        """
        Compute the center of each cell in the graph and build the image with contours and filled cells
        # param cells_groups: list of list: each list represent a group of cell that is going to be identify as such
        :return:
        """
        n_cells = len(self.coord)
        self.center_coord = list()
        if cells_to_hide is None:
            cells_to_hide = []

        self.cells_groups = cells_groups
        self.cells_groups_colors = cells_groups_colors

        img_contours = np.zeros((self.nb_lines, self.nb_col), dtype="int8")
        # key is the group, value a list of img
        special_cell_imgs = dict()
        non_special_cell_imgs = list()
        cells_in_groups = []
        if cells_groups is not None:
            for group_id, cells_group in enumerate(cells_groups):
                cells_in_groups.extend(cells_group)
                for special_cell in cells_group:
                    special_cell_img = np.zeros((self.nb_lines, self.nb_col), dtype="int8")

                    # early_born cell
                    c = self.coord[special_cell]
                    c = c - 1
                    c_filtered = c.astype(int)
                    # c = signal.medfilt(c)
                    special_cell_img[c_filtered[1, :], c_filtered[0, :]] = 1
                    morphology.binary_fill_holes(special_cell_img, output=special_cell_img)
                    # green value is -1
                    special_cell_img[c_filtered[1, :], c_filtered[0, :]] = 0
                    if group_id not in special_cell_imgs:
                        special_cell_imgs[group_id] = []
                    special_cell_imgs[group_id].append(special_cell_img)

        cells_in_groups = np.array(cells_in_groups)
        cells_not_in_groups = np.setdiff1d(np.arange(n_cells), cells_in_groups)

        for cell in cells_not_in_groups:
            non_special_cell_img = np.zeros((self.nb_lines, self.nb_col), dtype="int8")

            # early_born cell
            c = self.coord[cell]
            c = c - 1
            c_filtered = c.astype(int)
            # c = signal.medfilt(c)
            non_special_cell_img[c_filtered[1, :], c_filtered[0, :]] = 1
            morphology.binary_fill_holes(non_special_cell_img, output=non_special_cell_img)
            # green value is -1
            non_special_cell_img[c_filtered[1, :], c_filtered[0, :]] = 0
            non_special_cell_imgs.append(non_special_cell_img)
        # print(f"self.coord {self.coord}")

        for i, c in enumerate(self.coord):
            if i in cells_to_hide:
                continue

            # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
            c = c - 1

            # print(f"i {i}, c {c}")

            if c.shape[0] == 0:
                print(f'Error: {i} c.shape {c.shape}')
                continue
            # c = signal.medfilt(c)
            c_filtered = c.astype(int)
            bw = np.zeros((self.nb_lines, self.nb_col), dtype="int8")
            # morphology.binary_fill_holes(input
            bw[c_filtered[1, :], c_filtered[0, :]] = 1
            # early born as been drawn earlier, but we need to update center_coord
            img_contours[c_filtered[1, :], c_filtered[0, :]] = 1
            c_x, c_y = ndimage.center_of_mass(bw)
            self.center_coord.append((c_y, c_x))

        self.img_filled = np.zeros((self.nb_lines, self.nb_col), dtype="int16")
        # specifying output, otherwise binary_fill_holes return a boolean array
        # morphology.binary_fill_holes(img_contours, output=self.img_filled)
        # self.img_filled = self.img_filled * 2
        # self.img_filled[self.img_filled>0] = 2

        with_borders = False

        # now putting contour to value 0
        for i, c in enumerate(self.coord):
            if i in cells_to_hide:
                continue

            # it is necessary to remove one, as data comes from matlab, starting from 1 and not 0
            c = c - 1

            if c.shape[0] == 0:
                continue
            c_filtered = c.astype(int)
            # print(f"c_filtered {c_filtered}")
            # border to 2, to be in black
            if with_borders:
                self.img_filled[c_filtered[1, :], c_filtered[0, :]] = 2
            else:
                self.img_filled[c_filtered[1, :], c_filtered[0, :]] = 0

        if cells_groups_colors is not None:
            for group_id, special_cell_img_list in special_cell_imgs.items():
                # filling special cell with value to -1
                for special_cell_img in special_cell_img_list:
                    for n, pixels in enumerate(special_cell_img):
                        filled_pixels = np.where(pixels > 0)[0]
                        if len(filled_pixels) > 0:
                            # print(f"n {n}, filled_pixels {pixels[filled_pixels]}")
                            # print(f"group_id {group_id}")
                            self.img_filled[n, filled_pixels] = 4 + (2*group_id)

        for non_special_cell_img in non_special_cell_imgs:
            for n, pixels in enumerate(non_special_cell_img):
                filled_pixels = np.where(pixels > 0)[0]
                if len(filled_pixels) > 0:
                    # print(f"n {n}, filled_pixels {filled_pixels}")
                    self.img_filled[n, np.where(pixels > 0)[0]] = 2

        # if we dilate, some cells will fusion
        dilatation_version = False
        if (not with_borders) and dilatation_version:
            self.img_filled = morphology.binary_dilation(self.img_filled)

        self.img_contours = img_contours

    def plot_cells_map(self, param, data_id, title_option="", connections_dict=None,
                       background_color=(0, 0, 0, 1), default_cells_color=(1, 1, 1, 1.0),
                       link_connect_color="white", link_line_width=1,
                       cell_numbers_color="dimgray", show_polygons=False,
                       fill_polygons=True,
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

        fig, ax = plt.subplots(nrows=1, ncols=1,
                               gridspec_kw={'height_ratios': [1]},
                               figsize=(20, 20))
        # ax.set_facecolor("black")

        # blue = "cornflowerblue"
        # cmap.set_over('red')
        list_colors = [background_color, default_cells_color]
        bounds = [-3, 1, 3]
        if self.cells_groups is not None:
            for cells_groups_color in self.cells_groups_colors:
                list_colors.append(cells_groups_color)
                bounds.append(bounds[-1] + 2)
        # print(f"list_colors {list_colors}")
        cmap = mpl.colors.ListedColormap(list_colors)
        cmap.set_under('black')
        # cmap.set_over('black')
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        # norm = cm.colors.Normalize(vmax=abs(self.img_filled).max(), vmin=0)
        # print(f"self.img_filled: len {len(self.img_filled)}")
        # for n, pixels in enumerate(self.img_filled):
        #     if n == 75:
        #         print(f"n {n}: {list(self.img_filled[n , :])}")
        # print(f"bounds {bounds}, np.min(self.img_filled) {np.min(self.img_filled)},  "
        #       f"np.max(self.img_filled) {np.max(self.img_filled)}")
        # print(f"np.where(self.img_filled == 0) {len(np.where(self.img_filled == 0)[0])}")
        # print(f"np.where(self.img_filled == 2) {len(np.where(self.img_filled == 2)[0])}")
        for neuron, (c_x, c_y) in enumerate(cells_center):
            with_cells_img = True
            if with_cells_img:
                x, y = np.meshgrid(np.arange(0, len(self.img_filled), 1),
                                   np.arange(0, len(self.img_filled), 1))

                # for help: https://matplotlib.org/gallery/
                # images_contours_and_fields/
                # scontour_image.html#sphx-glr-gallery-images-contours-and-fields-contour-image-py
                cset1 = ax.contourf(x, y, self.img_filled, norm=norm,
                                    cmap=cm.get_cmap(cmap, len(bounds)-1)) # levels=bounds
                # for c in cset1.collections:
                #     c.set_edgecolor((0,0,0,0))
                # It is not necessary, but for the colormap, we need only the
                # number of levels minus 1.  To avoid discretization error, use
                # either this number or a large number such as the default (256).

                # If we want lines as well as filled regions, we need to call
                # contour separately; don't try to change the edgecolor or edgewidth
                # of the polygons in the collections returned by contourf.
                # Use levels output from previous call to guarantee they are the same.

                add_contours = False
                if add_contours:
                    cset2 = ax.contour(x, y, self.img_filled, levels=cset1.levels, colors="white")
                    for c in cset2.collections:
                        c.set_linewidth(0.5)

                # cset2 = ax.contour(x, y, self.img_filled, norm=norm,
                #                     cmap=cmap)

                # im1 = plt.imshow(self.img_filled, cmap=cmap, norm=norm)
                # # im1.set_zorder(5)

                # plt.contour(x, y,
                #             self.img_filled, colors=["white"],  linewidths=1,
                #             antialiased=True, zorder=1) # origin='image',
            else:
                color = "white"
                plt.scatter(x=c_x, y=c_y, marker='o', c=color, edgecolor="black", s=100, zorder=20)

            ylim = ax.get_ylim()
            # TODO: invert Y
            ax.set_ylim(ylim[::-1])

            fontsize = 6
            if neuron >= 100:
                fontsize = 4
            elif neuron >= 10:
                fontsize = 5

            zorder_lines = 15

            if with_cell_numbers:
                if with_cells_img:
                    plt.text(x=c_x, y=c_y,
                             s=f"{neuron}", color=cell_numbers_color, zorder=22,
                             ha='center', va="center", fontsize=fontsize + 2, fontweight='bold')
                else:
                    plt.text(x=c_x, y=c_y,
                             s=f"{neuron}", color=cell_numbers_color, zorder=22,
                             ha='center', va="center", fontsize=fontsize, fontweight='bold')

            if (connections_dict is not None) and (neuron in connections_dict):
                # plot a line to all out of the neuron
                for connected_neuron, nb_connexion in connections_dict[neuron].items():
                    line_width = link_line_width + np.log(nb_connexion)

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

        plt.title(f"Cells map {data_id} {title_option}")

        #  :param plot_option: if 0: plot n_out and n_int, if 1 only n_out, if 2 only n_in, if 3: only n_out with dotted to
        # show the commun n_in and n_out, if 4: only n_in with dotted to show the commun n_in and n_out,

        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{data_id}_cell_maps_{title_option}'
                        f'_{param.time_str}.{save_format}',
                        format=f"{save_format}")
        plt.close()


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
