using GraphNeuralNetworks, SparseArrays

# Create a toy graph with 5 nodes and 4 directed edges.
# The edge list defines connections: 1→2, 2→3, 3→4, 4→5.
# We represent the graph as a sparse adjacency matrix (5x5).
src = [1, 2, 3, 4]
dst = [2, 3, 4, 5]
adj = sparse(src, dst, ones(Float32, 4), 5, 5)

# Generate random node features:
# Each node has a 3-dimensional feature vector (3 × 5 matrix).
# This simulates a simple feature space where each column is a node.
x = rand(Float32, 3, 5)

# Define a single Graph Convolutional layer.
# This layer will transform 3-dimensional input features to 4-dimensional outputs.
gcn = GCNConv(3 => 4)

# Perform a forward pass through the GCN layer.
# The layer combines each node’s features with those of its neighbors,
# using the structure defined in the adjacency matrix.
out = gcn(x, adj)

# Output has shape (4, 5): 4 features per node, 5 nodes total.
@show size(out)
