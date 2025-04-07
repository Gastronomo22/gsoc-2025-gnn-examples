using GraphNeuralNetworks, SparseArrays

# Create a small graph with 5 nodes and 4 edges
src = [1, 2, 3, 4]
dst = [2, 3, 4, 5]
adj = sparse(src, dst, ones(Float32, 4), 5, 5)

# Node features: 3 features per node
x = rand(Float32, 3, 5)

# GCN layer: from 3 input features to 4 output features
gcn = GCNConv(3 => 4)

# Forward pass
out = gcn(x, adj)

@show size(out)  # should be (4, 5)

