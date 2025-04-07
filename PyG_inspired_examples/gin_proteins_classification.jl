using GraphNeuralNetworks, GraphSignals
using Flux, Statistics, Random

Random.seed!(42)

# Load dataset (graphs + labels)
dataset = load_proteins()
graphs = dataset.graphs
labels = dataset.targets

# Shuffle and split into train/test
n = length(graphs)
perm = randperm(n)
train_idx = perm[1:floor(Int, 0.8n)]
test_idx  = perm[floor(Int, 0.8n)+1:end]

# Define GIN model
function make_model(in_dim, out_dim)
    GIN(
        [
            Chain(Dense(in_dim, 32, relu), Dense(32, 32)),
            Chain(Dense(32, 32, relu), Dense(32, 32)),
            Chain(Dense(32, out_dim))
        ],
        sum_pool
    )
end

model = make_model(size(graphs[1].x, 1), length(unique(labels)))
loss_fn(g, y) = Flux.logitcrossentropy(model(g), y)
opt = ADAM(0.01)

# Training
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

# Evaluation
correct = 0
for i in test_idx
    g, y_true = graphs[i], labels[i]
    y_pred = Flux.onecold(model(g))
    correct += (y_pred == y_true)
end

accuracy = correct / length(test_idx)
@info "Test accuracy on PROTEINS: $accuracy"

