using GraphNeuralNetworks, GraphSignals
using Flux
using Statistics

# Load the Cora citation dataset.
# Nodes are papers; edges represent citations.
# Task: classify papers into subject categories.
dataset = load_cora()

# Extract graph structure and data
graph = dataset.graph                    # Contains adjacency and edge info
x = dataset.features                     # Node features (input for each paper)
y = dataset.targets                      # Ground truth labels (one per node)
train_mask = dataset.train_mask          # Boolean mask: true for training nodes
test_mask = dataset.test_mask            # Boolean mask: true for test nodes
adj = graph.adjacency                    # Sparse adjacency matrix (used by GAT)

# Define a two-layer Graph Attention Network (GAT).
# First layer: 8 attention heads each projecting to 8 features (output dim = 8×8).
# Second layer: single-head attention mapping to number of classes.
model = Chain(
    GATConv(size(x, 1) => 8, heads=8, relu),                     # Multi-head attention layer
    GATConv(8 * 8 => length(unique(y)), heads=1)                 # Final classification layer
)

# Loss function: multi-class cross entropy on training nodes only.
loss_fn(x, adj, y) = Flux.logitcrossentropy(model(x, adj)[:, train_mask], y[train_mask])

# Optimizer: basic gradient descent with small learning rate.
opt = Descent(0.005)

# Training loop: compute gradients and update model parameters over 100 epochs.
for epoch in 1:100
    grads = Flux.gradient(() -> loss_fn(x, adj, y), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn(x, adj, y))"
    end
end

# Evaluate the model on test nodes.
ŷ = model(x, adj)                                # Forward pass: get class scores for all nodes
preds = Flux.onecold(ŷ[:, test_mask])            # Convert scores to predicted class indices
acc = mean(preds .== y[test_mask])               # Compute test accuracy
@info "Test accuracy: $acc"
