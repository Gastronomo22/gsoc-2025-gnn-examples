using GraphNeuralNetworks, GraphSignals
using Flux, SparseArrays, Random, Statistics

Random.seed!(42)

# Load dataset
dataset = load_cora()
g = dataset.graph
x = dataset.features
adj = g.adjacency

# Extract positive edges (from upper triangle only to avoid duplicates)
pos_u, pos_v, _ = findnz(triu(adj))
n_edges = length(pos_u)

# Generate same number of negative edges
num_nodes = size(adj, 1)
neg_u = rand(1:num_nodes, n_edges)
neg_v = rand(1:num_nodes, n_edges)

# Define GraphSAGE model
model = Chain(
    GraphSAGEConv(size(x, 1) => 32, relu),
    GraphSAGEConv(32 => 32, relu)
)

# Score function: dot product between node embeddings
function edge_score(u, v)
    h = model(x, adj)
    return sum(h[:, u] .* h[:, v]; dims=1)
end

# Binary cross-entropy loss
function loss_fn()
    pos_scores = [edge_score(u, v) for (u, v) in zip(pos_u, pos_v)]
    neg_scores = [edge_score(u, v) for (u, v) in zip(neg_u, neg_v)]
    l_pos = Flux.logitbinarycrossentropy.(pos_scores, ones(length(pos_scores)))
    l_neg = Flux.logitbinarycrossentropy.(neg_scores, zeros(length(neg_scores)))
    return mean(vcat(l_pos, l_neg))
end

# Training
opt = ADAM(0.01)
for epoch in 1:50
    grads = Flux.gradient(() -> loss_fn(), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn())"
    end
end

