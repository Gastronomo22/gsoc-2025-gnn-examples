using GraphNeuralNetworks, GraphSignals
using Flux, Statistics, Random

Random.seed!(42)  # Ensure reproducibility of train/test split

# Load the PROTEINS dataset.
# Each graph represents a protein structure.
# Task: graph-level classification (e.g., whether a protein belongs to a certain class).
dataset = load_proteins()
graphs = dataset.graphs
labels = dataset.targets

# Shuffle and split indices into training (80%) and test (20%) sets
n = length(graphs)
perm = randperm(n)
train_idx = perm[1:floor(Int, 0.8n)]
test_idx  = perm[floor(Int, 0.8n)+1:end]

# Define a GIN model: uses a stack of MLPs (Dense layers) applied to node features.
# After message passing, the node embeddings are pooled via sum_pool to obtain a graph-level embedding.
function make_model(in_dim, out_dim)
    GIN(
        [
            Chain(Dense(in_dim, 32, relu), Dense(32, 32)),        # First MLP block
            Chain(Dense(32, 32, relu), Dense(32, 32)),            # Second MLP block
            Chain(Dense(32, out_dim))                             # Final output layer
        ],
        sum_pool  # Aggregation function to combine node features into graph representation
    )
end

# Build model with input dimension from node features and output equal to number of classes
model = make_model(size(graphs[1].x, 1), length(unique(labels)))

# Define loss function: cross-entropy for single graph
loss_fn(g, y) = Flux.logitcrossentropy(model(g), y)

# Optimizer
opt = ADAM(0.01)

# Training loop
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

# Evaluate on test set
correct = 0
for i in test_idx
    g, y_true = graphs[i], labels[i]
    y_pred = Flux.onecold(model(g))
    correct += (y_pred == y_true)
end

accuracy = correct / length(test_idx)
@info "Test accuracy on PROTEINS: $accuracy"
