# HPC Distributed Deep Learning Training Workflow Simulation with CUDA, MPI,  Nvidia RTX4070 GPU, & Rocky Linux (RHEL Compatible)

## Objective

#### In this project we will  simulate distributed AI training using MPI, Rocky Linux (RHEL compatible), Nvidia Geforce RTX 4070 GPU and CUDA.

#### Each MPI process:
    • Initializes a small neural network layer (a weight matrix).
    • Sends part of the training data to its assigned GPU.
    • Computes a matrix multiplication (A * B) — the core of neural network forward propagation.
    • Then uses MPI communication to aggregate (sum up) the results — similar to what happens in multi-GPU training (like PyTorch’s DistributedDataParallel).
    
#### This is similar to what frameworks like PyTorch Distributed, Horovod, or DeepSpeed do under the hood in HPC/AI environments.

---

##  Background Studies:

### Matrix Multiplication Is Everywhere in AI

#### Here are real-world examples:

| AI Operation | Internal Math | GPU Kernel Type |
|--------------|---------------|-----------------|
| Dense (Fully Connected) layer | Y=W×X+b | GEMM (General Matrix Multiply) |
| Convolution | Sliding window dot-products | Matrix multiply after "im2col" transform |
| Attention mechanism (Transformers) | QKᵀ and AV | Matrix multiplications |
| Backpropagation | Gradient = dL/dW=XᵀdY | Matrix multiplications again |

In fact, over 90% of the FLOPs (floating point operations) in models like GPT or ResNet are matrix multiplications.







