# gsoc-2025-gnn-examples
A collection of minimal and educational Julia examples using GraphNeuralNetworks.jl for my GSoC 2025 proposal, drawing inspiration from PyTorch Geometric (PyG) (https://pytorch-geometric.readthedocs.io)

The goal is to help new users understand how to apply GNN layers in Julia, by starting from simple examples and gradually reimplementing canonical PyG models and tasks.

Contributions, suggestions, and feedback are welcome!

# Examples

## introductory_examples

### `gcn_minimal.jl`
A minimal GCN example using random features and a sparse adjacency matrix.  
Useful for understanding how to apply a GCN layer without setup.

### `cora_node_classification.jl`
A simple GCN with two layers, trained on the Cora dataset for node classification.  
Includes training loop and test accuracy evaluation.

## PyG_inspired_examples



