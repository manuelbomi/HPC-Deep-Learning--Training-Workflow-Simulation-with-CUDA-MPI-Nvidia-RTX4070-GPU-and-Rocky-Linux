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

---

## Why GPUs and CUDA Matter for AI Training
#### GPUs are designed for massively parallel workloads — exactly what matrix multiplication needs.

    • A 256×256 matrix multiply has 65,536 output elements.
    • Each can be computed independently, making it ideal for GPUs.
    • CUDA essentially  launch thousands of threads that each compute one cell of the output matrix.
    
#### That is why GPUs dominate AI:

#### They are essentially matrix-multiplication machines.

---

## Why We Need MPI (Message Passing Interface)
#### MPI is a standardized API used in High-Performance Computing (HPC) for passing messages between processes, typically across:

- multiple CPUs

- multiple compute nodes

- multi-GPU or hybrid CPU/GPU clusters

#### It is the core communication layer used to build distributed training, scientific simulations, parallel numerical solvers, and more. MPI provides this communication layer — the “glue” that lets GPUs cooperate like one large distributed supercomputer.

#### When an AI model is trained on multiple GPUs or nodes:
    • Each GPU works on a portion of the data.
    • After computing gradients, all GPUs share and synchronize results using MPI.

#### This is why MPI is central to distributed CUDA jobs, multi-node Slurm workloads, and AI model parallelism.

#### MPI enables processes to exchange data using:

- MPI_Send, MPI_Recv → point-to-point communication

- MPI_Bcast, MPI_Reduce, MPI_Allreduce → collective communication

- MPI_Comm_rank, MPI_Comm_size → process management

- MPI_Barrier → synchronization

#### So combining:
- MPI + CUDA = scalable, distributed AI computation

#### That is exactly how clusters with H100s or A100s work in enterprise or research HPC environments.

---

## Our Contribution to HPC, GPU, CUDA & AI Model Training

#### Our approach and computations in this project are similar to the architecture pattern used in:

    • Supercomputers like Frontier, Perlmutter, Selene (NVIDIA’s own HPC AI cluster)
    • Cloud AI platforms (AWS P4d, Azure NDv4, Google TPU Pods)
    • Enterprise RHEL / Rocky Linux clusters running Slurm
    
#### The project here runs on an Nvidia Geforce RTX 4070 GPU,  but the approach is conceptually identical to what runs at petascale or exascale

#### Our Approach here showcase:
    • How HPC frameworks distribute workloads (MPI)
    • How GPUs handle neural network-style computation (CUDA)
    • How parallel processes communicate results (MPI Reduce)
    • How to prepare for Slurm cluster job submission later. For the slurm extension of this work, please check: 

---








