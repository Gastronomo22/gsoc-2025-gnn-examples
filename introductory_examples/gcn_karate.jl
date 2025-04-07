using GraphNeuralNetworks, GraphSignals
using Flux

# Load the Karate Club dataset.
# This is a small social network where nodes represent members of a karate club,
# and edges represent social interactions. The goal is to classify nodes into communities.
dataset = load_karateclub()

# Extract graph structure and data
g = dataset.graph                # Graph object (contains adjacency, edge index, etc.)
x = dataset.features             # Node features: (num_features × num_nodes)
y = dataset.targets              # Ground truth class labels for each node
adj = g.adjacency                # Sparse adjacency matrix of the graph

# Define a two-layer Graph Convolutional Network (GCN).
# The first layer projects input features to a hidden space and applies ReLU activation.
# The second layer maps hidden features to logits over the class labels.
model = Chain(
    GCNConv(size(x, 1) => 16, relu),
    GCNConv(16 => length(unique(y)))
)

# Loss function: multi-class cross entropy on all nodes.
# Since this is a small dataset, we train on the full graph without a mask.
loss_fn() = Flux.logitcrossentropy(model(x, adj), y)

# Optimizer: simple gradient descent with fixed learning rate.
opt = Descent(0.01)

# Training loop: compute gradients and update weights over 100 epochs.
for epoch in 1:100
    grads = Flux.gradient(() -> loss_fn(), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
end

# Evaluate performance.
ŷ = model(x, adj)                    # Forward pass: compute logits for all nodes
preds = Flux.onecold(ŷ)             # Convert logits to predicted class indices
accuracy = sum(preds .== y) / length(y)   # Compute accuracy over all nodes
@info "Test accuracy on Karate Club: $accuracy"
