using GraphNeuralNetworks, GraphSignals
using Flux, SparseArrays, Random, Statistics

Random.seed!(42)  # Ensure reproducibility

# Load the Cora dataset.
# Nodes represent papers; edges are citations.
# We'll use the graph structure for link prediction, ignoring node labels.
dataset = load_cora()
g = dataset.graph
x = dataset.features
adj = g.adjacency

# Extract positive edges from the graph.
# We only take the upper triangle of the adjacency matrix to avoid counting both (u,v) and (v,u).
pos_u, pos_v, _ = findnz(triu(adj))  # u and v are lists of node indices
n_edges = length(pos_u)

# Generate an equal number of negative edges (i.e., node pairs that are not connected).
# These are used as negative samples for binary classification.
num_nodes = size(adj, 1)
neg_u = rand(1:num_nodes, n_edges)
neg_v = rand(1:num_nodes, n_edges)

# Define a 2-layer GraphSAGE model.
# GraphSAGE aggregates neighborhood features and supports inductive learning.
# We'll use it to compute node embeddings.
model = Chain(
    GraphSAGEConv(size(x, 1) => 32, relu),   # Hidden layer
    GraphSAGEConv(32 => 32, relu)            # Output embedding layer
)

# Define a score function for edge prediction.
# For a pair of nodes (u, v), the dot product of their embeddings is used as a score.
function edge_score(u, v)
    h = model(x, adj)                            # Get embeddings for all nodes
    return sum(h[:, u] .* h[:, v]; dims=1)       # Element-wise product + sum = dot product
end

# Define binary classification loss using positive and negative edge pairs.
# For each positive edge, the score should be high (close to 1).
# For each negative edge, the score should be low (close to 0).
function loss_fn()
    pos_scores = [edge_score(u, v) for (u, v) in zip(pos_u, pos_v)]
    neg_scores = [edge_score(u, v) for (u, v) in zip(neg_u, neg_v)]

    l_pos = Flux.logitbinarycrossentropy.(pos_scores, ones(length(pos_scores)))
    l_neg = Flux.logitbinarycrossentropy.(neg_scores, zeros(length(neg_scores)))

    return mean(vcat(l_pos, l_neg))  # Combine and average all losses
end

# Training loop: standard backpropagation using ADAM optimizer
opt = ADAM(0.01)
for epoch in 1:50
    grads = Flux.gradient(() -> loss_fn(), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn())"
    end
end
