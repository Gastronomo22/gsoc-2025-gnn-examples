# gsoc-2025-gnn-examples
A collection of minimal and educational Julia examples using GraphNeuralNetworks.jl for my GSoC 2025 proposal, drawing inspiration from [PyTorch Geometric (PyG)](https://pytorch-geometric.readthedocs.io).

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

### `karate_gcn_minimal.jl`
This example demonstrates a basic two-layer Graph Convolutional Network (GCN) applied to the well-known Karate Club dataset.  
Due to its simplicity, the dataset allows for fast training and easy visualization, making it a great starting point for understanding node classification with GNNs.

## PyG_inspired_examples

### `gat_cora.jl`
Reimplementation of the classic Graph Attention Network (GAT) model on the Cora dataset.  
This example uses two GATConv layers with 8 attention heads and a simple Flux training loop.

### `gin_mutag.jl`
GIN model for graph classification on the MUTAG dataset, adapted from PyTorch Geometric’s `gin.py`.  
Uses sum pooling and a simple training loop to classify whole graphs.

### `sage_link_prediction.jl`
A PyG-inspired example that uses GraphSAGE to perform link prediction on the Cora dataset.  
The model learns to predict whether an edge exists between two nodes, using dot-product scoring on learned embeddings.  
Positive and negative samples are generated from the graph structure, and training is done with binary cross-entropy loss.

### `gin_proteins_classification.jl`
A graph classification example using the Graph Isomorphism Network (GIN) on the PROTEINS dataset.  
This script demonstrates how to classify entire graphs using sum pooling and a simple training loop.  
Inspired by PyG's `gin.py` example, adapted for GraphNeuralNetworks.jl.





