using GraphNeuralNetworks, GraphSignals
using Flux
using Statistics
using Random

Random.seed!(42)  # Ensure reproducibility of shuffling

# Load MUTAG dataset.
# Each example is a separate molecular graph labeled by mutagenic effect.
# Task: graph-level classification (binary).
dataset = load_mutag()
graphs = dataset.graphs         # List of individual graphs
labels = dataset.targets        # One label per graph

# Shuffle and split indices into 80% training, 20% test
n = length(graphs)
perm = randperm(n)
train_idx = perm[1:floor(Int, 0.8n)]
test_idx  = perm[floor(Int, 0.8n)+1:end]

# Define a GIN (Graph Isomorphism Network) model with sum pooling.
# This model applies multiple MLP layers (Dense) to node features,
# then pools node embeddings to produce a graph-level embedding.
function make_model(in_dim, out_dim)
    GIN(
        [ Dense(in_dim, 32, relu),
          Dense(32, 32),
          Dense(32, out_dim) ],
        sum_pool    # Aggregate all node embeddings into a single graph embedding
    )
end

# Build model: input dimension from node features, output = num classes
model = make_model(size(graphs[1].x, 1), length(unique(labels)))

# Loss: multi-class cross entropy on one graph
loss_fn(g, y) = Flux.logitcrossentropy(model(g), y)
opt = ADAM(0.01)

# Training loop: full batch (graph-by-graph)
for epoch in 1:100
    total_loss = 0.0
    for i in train_idx
        g, y_true = graphs[i], labels[i]
        grads = Flux.gradient(() -> loss_fn(g, y_true), Flux.params(model))
        Flux.Optimise.update!(opt, Flux.params(model), grads)
        total_loss += loss_fn(g, y_true)
    end
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss = $(total_loss / length(train_idx))"
    end
end

# Evaluation on test set: count correct predictions
correct = 0
for i in test_idx
    g, y_true = graphs[i], labels[i]
    y_pred = Flux.onecold(model(g))
    correct += (y_pred == y_true)
end
accuracy = correct / length(test_idx)
@info "Test accuracy: $accuracy"
