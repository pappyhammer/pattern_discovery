import networkx as nx
# from fa2 import ForceAtlas2
import matplotlib.pyplot as plt


def plot_graph_using_fa2(graph, file_name="", param=None, iterations=2000, save_raster=True,
                         with_labels=True, title=None,
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
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 10))
    fig.tight_layout()
    ax.set_facecolor("black")
    nx.draw_networkx(graph, pos=positions, node_size=10, edge_color="white",
                     node_color="cornflowerblue", arrowsize=4, width=0.4,
                     with_labels=with_labels, arrows=True,
                     ax=ax)
    # nx.draw_networkx(graph, node_size=10, edge_color="white",
    #                  node_color="cornflowerblue",
    #                  with_labels=with_labels, arrows=True,
    #                  ax=ax)

    if title is not None:
        plt.title(title)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    if show_plot:
        plt.show()

    if save_raster and (param is not None):
        # transforming a string in a list
        if isinstance(save_formats, str):
            save_formats = [save_formats]
        for save_format in save_formats:
            fig.savefig(f'{param.path_results}/{file_name}_{param.time_str}.{save_format}', format=f"{save_format}")

    plt.close()
