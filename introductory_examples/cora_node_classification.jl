using GraphNeuralNetworks, GraphSignals
using Flux
using Statistics

# Load dataset
dataset = load_cora()
graph = dataset.graph
x = dataset.features
y = dataset.targets
train_mask = dataset.train_mask
test_mask = dataset.test_mask
adj = graph.adjacency

# Define 2-layer GCN
model = Chain(
    GCNConv(size(x, 1) => 16, relu),
    GCNConv(16 => length(unique(y)))
)

loss_fn(x, adj, y) = Flux.logitcrossentropy(model(x, adj)[:, train_mask], y[train_mask])

# Train loop
opt = Descent(0.01)
for epoch in 1:100
    grads = Flux.gradient(() -> loss_fn(x, adj, y), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn(x, adj, y))"
    end
end

# Evaluate
ŷ = model(x, adj)
preds = Flux.onecold(ŷ[:, test_mask])
acc = mean(preds .== y[test_mask])
@info "Test accuracy: $acc"

