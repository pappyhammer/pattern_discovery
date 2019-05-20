import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


def plot_graph_using_fa2(graph, file_name="", param=None, iterations=2000, save_raster=True,
                         color=None,
                         with_labels=True, title=None, ax_to_use=None,
                         save_formats="pdf", show_plot=False):

    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=False,  # Dissuade hubs
        linLogMode=False,  # NOT IMPLEMENTED
        adjustSizes=False,  # Prevent overlap (NOT IMPLEMENTED)
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=1.0,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.2,
        multiThreaded=False,  # NOT IMPLEMENTED

        # Tuning
        scalingRatio=3.0,
        strongGravityMode=False,
        gravity=1.0,

        # Log
        verbose=True)

    # to open with Cytoscape
    nx.write_graphml(graph, f"{param.path_results}/{file_name}.graphml")
    nx.write_gexf(graph, f"{param.path_results}/{file_name}.gexf")


    positions = forceatlas2.forceatlas2_networkx_layout(graph, pos=None, iterations=iterations)

    if ax_to_use is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
        fig.tight_layout()
        ax.set_facecolor("black")
    else:
        ax = ax_to_use
    if color is None:
        color = "cornflowerblue"
    nx.draw_networkx(graph, pos=positions, node_size=10, edge_color="white",
                     node_color=color, arrowsize=4, width=0.4,
                     with_labels=with_labels, arrows=True,
                     ax=ax)
    # nx.draw_networkx(graph, node_size=10, edge_color="white",
    #                  node_color="cornflowerblue",
    #                  with_labels=with_labels, arrows=True,
    #                  ax=ax)
    if ax_to_use is not None:
        legend_elements = []
        legend_elements.append(Patch(facecolor=color,
                                     edgecolor='white', label=f'{title}'))
        ax.legend(handles=legend_elements)

    if (title is not None) and (ax_to_use is None):
        plt.title(title)

    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

    if ax_to_use is None:
        if show_plot:
            plt.show()

        if save_raster and (param is not None):
            # transforming a string in a list
            if isinstance(save_formats, str):
                save_formats = [save_formats]
            for save_format in save_formats:
                fig.savefig(f'{param.path_results}/{file_name}_{param.time_str}.{save_format}', format=f"{save_format}")

        plt.close()
