using GraphNeuralNetworks, GraphSignals
using Flux, Statistics

# Load the Cora dataset.
# Each node is a paper; edges are citation links.
# The task is to classify papers by subject.
dataset = load_cora()

# Extract components
graph = dataset.graph
x = dataset.features
y = dataset.targets
train_mask = dataset.train_mask
test_mask = dataset.test_mask
adj = graph.adjacency

# Define a model using APPNPConv.
# APPNP applies personalized PageRank to propagate predictions through the graph.
# alpha = teleport probability, k = number of propagation steps
model = Chain(
    Dense(size(x, 1), 64, relu),                     # Standard MLP layer
    Dense(64, length(unique(y))),                    # Final class logits
    APPNPConv(k=10, alpha=0.1)                       # PageRank-style propagation
)

# Loss: cross-entropy over training nodes only
loss_fn() = Flux.logitcrossentropy(model(x, adj)[:, train_mask], y[train_mask])

# Optimizer
opt = ADAM(0.01)

# Training loop
for epoch in 1:100
    grads = Flux.gradient(() -> loss_fn(), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn())"
    end
end

# Evaluation
ŷ = model(x, adj)
preds = Flux.onecold(ŷ[:, test_mask])
accuracy = mean(preds .== y[test_mask])
@info "Test accuracy with APPNP: $accuracy"
