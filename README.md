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
    • How to prepare for Slurm cluster job submission later. For the slurm extension of this work, please check: https://github.com/manuelbomi/End-to-End-HPC-AI-Training-Simulation-with-CUDA-MPI-and-Slurm-on-Rocky-Linux-----RHEL-Compatible

---

## Step 1 — Write the CUDA + MPI Code

#### On Rocky Linux, we create a file named mpi_cuda_matrix_mul.cu using nano (not cat).

```python
nano ~/mpi_cuda_matrix_mul.cu
```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/c80b138b-237a-432d-ab03-9df78938ea57" />

#### <ins>mpi_cuda_matrix_mul.cu</ins> has the CUDA code below:

```python

#include <mpi.h>
#include <stdio.h>
#include <cuda_runtime.h>

// -----------------------------
// CUDA KERNEL: Matrix Multiply
// -----------------------------
// Each thread computes one element of the resulting matrix C
// C = A * B
__global__ void matrixMul(const float *A, const float *B, float *C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

// -----------------------------
// MAIN PROGRAM
// -----------------------------
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Determine number of GPUs and assign one per MPI process
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(world_rank % deviceCount);

    const int N = 256;  // Matrix size (N x N)
    const size_t bytes = N * N * sizeof(float);

    // Allocate and initialize host matrices
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Fill matrices A and B with simple patterns
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f + world_rank;  // Different value per process
        h_B[i] = 0.5f;
    }

    // Allocate GPU memory
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy matrices to GPU
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define CUDA grid and block dimensions
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + 15) / 16, (N + 15) / 16);

    // Launch CUDA kernel for matrix multiplication
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Compute local sum of results (simulating model aggregation)
    float local_sum = 0.0f;
    for (int i = 0; i < N * N; i++)
        local_sum += h_C[i];

    // Reduce (sum) all results from all processes into process 0
    float global_sum = 0.0f;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (world_rank == 0) {
        printf("\n[HPC Simulation] Matrix size: %dx%d\n", N, N);
        printf("[HPC Simulation] Total processes: %d\n", world_size);
        printf("[Result] Aggregated global sum of outputs: %.2f\n", global_sum);
        printf("[Status] Simulation complete on %d GPU(s).\n\n", deviceCount);
    }

    // Clean up
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);

    MPI_Finalize();
    return 0;
}

```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/412b7f92-82ef-4879-9920-bb343180935f" />

#### Save and give file the necessary permission

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/c06204a6-c3e7-48d7-b8f0-f1484c5eddfc" />

---

## Step 2 — Compile with NVCC + MPI

#### We use <ins>nvcc -ccbin mpicxx</ins> so that NVCC uses MPI’s C++ compiler as the host compiler.

```python
nvcc -ccbin mpicxx ~/mpi_cuda_matrix_mul.cu -o ~/mpi_cuda_matrix_mul
```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/33153809-369b-46c6-8f69-e21d1495c0c9" />

---

## Step 3 — Run with Multiple Processes

#### Run with 2 or more MPI processes:

```python
mpirun -np 2 ~/mpi_cuda_matrix_mul
```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/d091f132-f325-42f1-8488-f91dc2354e31" />

#### Example Output:

```python
[HPC Simulation] Matrix size: 256x256
[HPC Simulation] Total processes: 2
[Result] Aggregated global sum of outputs: 8388608.00
[Status] Simulation complete on 1 GPU(s).

```

<img width="1600" height="900" alt="Image" src="https://github.com/user-attachments/assets/e9f4c349-1aa8-4b22-9b45-01c32af2510b" />

---

## Similarity of Our Approach to Real-world HPC AI systems:

| Concept | Real Cluster Equivalent | What Happens Here |
|---------|------------------------|-------------------|
| MPI process | Worker node / GPU | Each process simulates a node computing its own batch |
| CUDA kernel | GPU computation | Each GPU performs matrix multiply (like forward pass) |
| MPI Reduce | Gradient aggregation | Combines all results like gradient averaging |
| CUDA memory | GPU VRAM | Data is copied in/out of GPU memory manually |
| NVCC + MPI | Training framework backend | Equivalent to Horovod or PyTorch distributed backend |

#### The table below connect the pieces in our CUDA+MPI code shared in this project to a real AI training job:

| In Our Code | In Deep Learning | Explanation |
|-------------|------------------|-------------|
| `matrixMul<<<>>>` | Forward pass (e.g., W * X) | Each GPU computes local tensor results |
| `MPI_Reduce` | Gradient aggregation / all-reduce | Combines partial results (synchronizes models) |
| `cudaMemcpy` | Data movement between GPU and CPU | Similar to how tensors move between host and device memory |
| `MPI_Init` / `MPI_Finalize` | Training process coordination | Equivalent to distributed PyTorch or Horovod initialization |
| Multiple ranks (`-np 2`) | Multi-GPU training nodes | Each process simulates a GPU worker |

---

## How to Improve on this project

| Goal | Description |
|------|-------------|
| Scale up N | Increase matrix size to test performance scaling. |
| Add more MPI processes | Run with `-np 4` to simulate a 4-GPU cluster. |
| Add random initialization | Replace constants with random weights. |
| Loop training iterations | Add multiple passes to simulate epochs. |
| Benchmark | Measure computation and communication time with `MPI_Wtime()`. |

---




## Summary

#### This project simulates:
    • A distributed deep learning workflow.
    • Real HPC GPU operations under MPI coordination.
    • A cluster-like workflow on a single system using Rocky Linux.
    
#### The ideas shared in this project are similar to using CUDA, MPI, and HPC workflows pn Nvidia H100/A100 clusters and integrate later with Slurm job schedulers.

## Slurm + CUDA + MPI based version of the project
#### For completeness, interested readers may wish to check the Slurm based version of the project. It is available here:  https://github.com/manuelbomi/End-to-End-HPC-AI-Training-Simulation-with-CUDA-MPI-and-Slurm-on-Rocky-Linux-----RHEL-Compatible.git





