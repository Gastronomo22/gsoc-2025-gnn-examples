using GraphNeuralNetworks, GraphSignals
using Flux, Statistics

# Load dataset (Cora)
dataset = load_cora()
g = dataset.graph
x = dataset.features
y = dataset.targets
train_mask = dataset.train_mask
test_mask = dataset.test_mask
adj = g.adjacency

# Define GAT model with 8 heads
model = Chain(
    GATConv(size(x, 1) => 8, heads=8, relu),  # Output: 8*8 = 64
    GATConv(64 => length(unique(y)), heads=1) # Final classification layer
)

loss_fn() = Flux.logitcrossentropy(model(x, adj)[:, train_mask], y[train_mask])
opt = ADAM(0.01)

# Training loop
for epoch in 1:100
    grads = Flux.gradient(() -> loss_fn(), Flux.params(model))
    Flux.Optimise.update!(opt, Flux.params(model), grads)
    if epoch % 10 == 0
        @info "Epoch $epoch | Loss: $(loss_fn())"
    end
end

# Evaluate
ŷ = model(x, adj)
preds = Flux.onecold(ŷ[:, test_mask])
accuracy = mean(preds .== y[test_mask])
@info "Test accuracy on Cora (GAT): $accuracy"

