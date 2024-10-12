import networkx as nx
import matplotlib.pyplot as plt

# Load the graph from the .graphml file
file_path = 'Social_Graph.graphml'  # Update with the path to your .graphml file
graph = nx.read_graphml(file_path)

# Fixing the position of nodes to maintain consistent layout
pos = nx.spring_layout(graph)

# Define nodes and edges to highlight
highlight_nodes = ['1', '2']  # Update with your node identifiers
highlight_edges = [('1', '3'), ('2', '4')]  # Update with your edge pairs

# Plot all nodes and edges with default colors
nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_color='lightblue', edge_color='gray')

# Highlight specified nodes and edges
nx.draw_networkx_nodes(graph, pos, nodelist=highlight_nodes, node_color='yellow')
nx.draw_networkx_edges(graph, pos, edgelist=highlight_edges, edge_color='yellow')

plt.title('Social Graph with Highlights')
plt.show()
