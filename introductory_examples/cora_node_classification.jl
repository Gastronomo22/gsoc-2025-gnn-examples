using GraphNeuralNetworks, GraphSignals
using Flux
using Statistics

# Load the Cora citation network dataset.
# Each node is a paper, edges are citation links. Task: node classification.
dataset = load_cora()

# Extract components from the dataset:
graph = dataset.graph               # Graph structure (edge list, adjacency, etc.)
x = dataset.features                # Node features: size (num_features, num_nodes)
y = dataset.targets                 # Integer labels per node
train_mask = dataset.train_mask    # Boolean mask: true for training nodes
test_mask = dataset.test_mask      # Boolean mask: true for test nodes
adj = graph.adjacency              # Sparse adjacency matrix of the graph

# Define a two-layer GCN model.
# The first GCNConv layer maps input features to a hidden space and applies ReLU.
# The second layer maps hidden representations to class logits.
model = Chain(
    GCNConv(size(x, 1) => 16, relu),
    GCNConv(16 => length(unique(y)))
)

# Define the loss function using logitcrossentropy on training nodes only.
# The model output is restricted to training nodes via boolean indexing.
loss_fn(x, adj, y) = Flux.logitcrossentropy(model(x, adj)[:, train_mask], y[train_mask])

# Use simple gradient descent optimizer for training.
opt = Descent(0.01)

# Training loop: 100 epochs
for epoch in 1:100
    grads = Flux.gradient(() -> loss_fn(x, adj, y), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)

    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn(x, adj, y))"
    end
end

# Evaluate model performance on test nodes.
ŷ = model(x, adj)                          # Forward pass on all nodes
preds = Flux.onecold(ŷ[:, test_mask])     # Convert logits to class predictions
acc = mean(preds .== y[test_mask])        # Accuracy: proportion of correct predictions
@info "Test accuracy: $acc"
