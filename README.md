## 1. Overview

![Scheme](img/1.png)

Matrix **A** is partitioned by rows among the MPI processes/GPUs, while matrix **B** must be available to all GPUs.

Each process computes a local portion of matrix **C**:

$$
C_i = A_i \times B
$$

where:

* \($A_i$\) is the portion of matrix **A** assigned to process/GPU \(i\);
* \($B$\) is the complete matrix **B**;
* \($C_i$\) is the partial result computed by process/GPU \(i\).

The partial matrices \($C_i$\) are then combined to obtain the complete result matrix:

$$
C =
\begin{bmatrix}
C_0 \\
C_1 \\
\vdots \\
C_{P-1}
\end{bmatrix}
$$

where \($P$\) is the number of MPI processes, GPUs, and NVSHMEM Processing Elements (PEs).

---

## 2. Parallel Execution Model

The application follows a **one MPI process per GPU** execution model.

Each MPI rank is associated with:

1. one CUDA device;
2. one NVSHMEM Processing Element (PE);
3. one participant in the NCCL communicator.

Conceptually:

```text
MPI Rank 0  --->  NVSHMEM PE 0  --->  GPU 0
MPI Rank 1  --->  NVSHMEM PE 1  --->  GPU 1
MPI Rank 2  --->  NVSHMEM PE 2  --->  GPU 2
     ...              ...              ...
MPI Rank P  --->  NVSHMEM PE P  --->  GPU P
```

MPI initializes and coordinates the distributed processes.

NVSHMEM is initialized using the existing MPI communicator:

```cpp
nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM,
                   &attr
);
```

The implementation verifies that MPI ranks and NVSHMEM PEs have a consistent mapping:

```text
MPI Rank i == NVSHMEM PE i
```

NCCL is also initialized across the GPUs associated with the MPI ranks.

This architecture allows the same GPU computational kernel to be evaluated with four different communication mechanisms.

---

## 3. Communication Libraries

The communication mechanism used for matrices **A**, **B**, and **C** can be independently selected using a three-character configuration string.

The available options are:

| Code | Communication Mechanism |
| ---- | ----------------------- |
| `M`  | MPI                     |
| `C`  | CUDA-Aware MPI          |
| `N`  | NCCL                    |
| `S`  | NVSHMEM                 |

The three positions in the configuration string correspond to:

```text
communication_libraries[0] -> distribution of matrix A
communication_libraries[1] -> broadcast/distribution of matrix B
communication_libraries[2] -> collection of matrix C
```

For example:

```text
SNS
```

means:

```text
Matrix A  -> NVSHMEM
Matrix B  -> NCCL
Matrix C  -> NVSHMEM
```

Other possible configurations include:

```text
MMM
CCC
NNN
SSS

MCS
NCS
SNS
SNC
MSN
```

This design makes it possible to evaluate different combinations of communication mechanisms without modifying the computational kernel.

---

## 4. Execution Flow

The general execution flow is:

```text
                 Matrix A                    Matrix B
                    |                           |
                    v                           v
             Distribution                  Distribution
             M / C / N / S                 M / C / N / S
                    |                           |
                    +------------+--------------+
                                 |
                                 v
                         Multiple GPUs / PEs
                                 |
                                 v
                         GPU_i: C_i = A_i x B
                                 |
                                 v
                           Collection of C
                            M / C / N / S
```

The computation can be divided into three main communication stages.

### Stage 1 — Distribution of Matrix A

Matrix **A** is partitioned among the GPUs.

Each GPU receives only the rows required to calculate its corresponding portion of matrix **C**.

### Stage 2 — Distribution of Matrix B

Matrix **B** must be available to every GPU participating in the computation.

Therefore, the complete matrix **B** is distributed to all participating processes/GPUs.

### Stage 3 — Collection of Matrix C

After each GPU computes its local matrix:

$$
C_i = A_i \times B
$$

the partial results are collected to construct the complete matrix **C**.

---

## 5. Communication Operations

### Matrix A

The distribution of matrix **A** depends on the selected communication mechanism:

| Option | Operation                                                    |
| ------ | ------------------------------------------------------------ |
| `M`    | `MPI_Scatter` using host memory                              |
| `C`    | `MPI_Scatter` directly using GPU memory                      |
| `N`    | `ncclBroadcast` followed by selection of the local partition |
| `S`    | `nvshmem_double_get` from PE 0                               |

For NVSHMEM, each PE performs a one-sided read of its corresponding portion of matrix **A**:

```cpp
nvshmem_double_get(s_lA, 
                   s_A + (size_t)myPE * mi * k, 
                   (size_t)mi * k, 
                   0
);
```

Therefore, each PE directly obtains its required block from the symmetric memory belonging to **PE 0**.

---

### Matrix B

Matrix **B** must be available to all participating GPUs:

| Option | Operation                             |
| ------ | ------------------------------------- |
| `M`    | `MPI_Bcast` using host memory         |
| `C`    | `MPI_Bcast` directly using GPU memory |
| `N`    | `ncclBroadcast`                       |
| `S`    | `nvshmem_double_get` from PE 0        |

For NVSHMEM, every non-root PE retrieves the complete matrix **B** from PE 0:

```cpp
if (myPE != 0)
    nvshmem_double_get(s_B,
                       s_B,
                       (size_t)k * n,
                       0
    );
```

---

### Matrix C

The partial results are combined using:

| Option | Operation                                                    |
| ------ | ------------------------------------------------------------ |
| `M`    | `MPI_Gather` using host memory                               |
| `C`    | `MPI_Gather` directly using GPU memory                       |
| `N`    | `ncclAllGather`                                              |
| `S`    | PE 0 retrieves the local C blocks using `nvshmem_double_get` |

With NVSHMEM, PE 0 obtains the partial result from every PE:

```cpp
if (myPE == 0)
{
    for (int pe = 0; pe < nPEs; ++pe)
    {
        nvshmem_double_get(s_C + (size_t)pe * mi * n,
                           s_lC,
                           (size_t)mi * n,
                           pe
        );
    }
}
```

The complete matrix **C** is therefore assembled in the symmetric memory of PE 0.

---

## 6. NVSHMEM Symmetric Memory

One of the main differences introduced by the NVSHMEM implementation is the use of **symmetric memory**.

The following symmetric GPU buffers are allocated:

```cpp
double *s_A;
double *s_B;
double *s_C;
double *s_lA;
double *s_lC;
```

using:

```cpp
nvshmem_malloc()
```

Conceptually:

```text
                 NVSHMEM Symmetric Heap

PE 0 / GPU 0       PE 1 / GPU 1       PE 2 / GPU 2
+-----------+      +-----------+      +-----------+
|   s_A     |      |   s_A     |      |   s_A     |
|   s_B     |      |   s_B     |      |   s_B     |
|   s_C     |      |   s_C     |      |   s_C     |
|   s_lA    |      |   s_lA    |      |   s_lA    |
|   s_lC    |      |   s_lC    |      |   s_lC    |
+-----------+      +-----------+      +-----------+
      ^                  ^                  ^
      |                  |                  |
      +-------- NVSHMEM one-sided ----------+
                communication
```

The symmetric memory model allows a PE to directly access memory associated with another PE using NVSHMEM communication operations.

This differs from the traditional MPI two-sided communication model, where both sender and receiver participate explicitly in the communication operation.

---

## 7. NVSHMEM Communication Model

The NVSHMEM implementation primarily uses:

```cpp
nvshmem_double_get()
```

This operation follows a **one-sided communication model**.

Conceptually:

```text
Traditional MPI

PE 0                         PE 1
 |                            |
 | MPI_Send                   |
 | -------------------------> |
 |                            | MPI_Recv
 |                            |
```

With NVSHMEM:

```text
NVSHMEM

PE 0                         PE 1
 |                            |
 |       symmetric data       |
 |                            |
 | <------ GET issued --------|
 |                            |
```

The requesting PE initiates the transfer directly.

For matrix **A**:

```text
PE i
 |
 | nvshmem_double_get()
 v
PE 0: s_A
 |
 v
PE i: s_lA
```

For matrix **B**:

```text
PE i
 |
 | nvshmem_double_get()
 v
PE 0: s_B
 |
 v
PE i: s_B
```

For matrix **C**, PE 0 retrieves the results:

```text
PE 0
 |
 +---- GET ---- PE 0:s_lC
 |
 +---- GET ---- PE 1:s_lC
 |
 +---- GET ---- PE 2:s_lC
 |
 +---- GET ---- PE N:s_lC
 |
 v
PE 0:s_C
```

---

## 8. Synchronization

NVSHMEM synchronization is performed using:

```cpp
nvshmem_barrier_all();
```

For example, after PE 0 initializes matrices **A** and **B**:

```cpp
nvshmem_barrier_all();
```

ensures that initialization is complete before other PEs attempt to retrieve the data.

Synchronization is also performed after NVSHMEM communication operations where required by the current implementation.

CUDA synchronization is performed using:

```cpp
cudaDeviceSynchronize();
```

and NCCL operations are synchronized through:

```cpp
cudaStreamSynchronize(s);
```

---

## 9. GPU Computation

The local matrix multiplication is performed by:

```cpp
ABMultiply()
```

which is defined in:

```text
mulmat_kernel.cu
```

The CUDA implementation uses **tiled matrix multiplication**.

Tiles from matrices **A** and **B** are loaded into GPU shared memory and multiplied to calculate the corresponding tile of matrix **C**.

Conceptually:

```text
Global GPU Memory
       |
       +---- Tile A ----+
       |                |
       +---- Tile B ----+
                |
                v
           Shared Memory
                |
                v
         Tile Multiplication
                |
                v
            Tile of C
```

The tile dimension is defined as:

```cpp
#define TILE_DIM 32
```

The kernel uses two shared-memory tiles:

```cpp
double *aTile;
double *bTile;
```

and the CUDA kernel is launched by:

```cpp
sharedABMultiply<<<grid, block, 2*w*w*sizeof(double)>>>(a, b, c, m, n, k, lda, ldb, ldc, w);
```

---

## 10. Selecting the Buffers for the CUDA Kernel

The same CUDA matrix multiplication kernel is used regardless of the selected communication mechanism.

The input and output buffers are dynamically selected according to the configuration:

```cpp
double *kernel_A = (communication_libraries[0] == 'S') ? s_lA : d_lA;

double *kernel_B = (communication_libraries[1] == 'S') ? s_B : d_B;

double *kernel_C = (communication_libraries[2] == 'S') ? s_lC : d_lC;
```

The multiplication is then executed as:

```cpp
ABMultiply(kernel_A,
           kernel_B,
           kernel_C,
           mi, 
           n, 
           k,
           lda, 
           ldb, 
           ldc,
           w
);
```



---

## 11. Source Files

The main source files are:

```text
.
├── mmb_nvshmem.cu
└── mulmat_kernel.cu
```

### `mmb_nvshmem.cu`

Implements:

* MPI initialization;
* GPU selection;
* NVSHMEM initialization;
* MPI Rank / NVSHMEM PE mapping;
* NCCL communicator initialization;
* host-memory allocation;
* CUDA device-memory allocation;
* NVSHMEM symmetric-memory allocation;
* matrix initialization;
* communication mechanism selection;
* distribution of matrix A;
* distribution of matrix B;
* invocation of the CUDA matrix multiplication kernel;
* collection of matrix C;
* execution-time measurement;
* resource cleanup.

### `mulmat_kernel.cu`

Implements the CUDA kernels responsible for matrix multiplication, including:

```cpp
simpleMultiply()
```

and the tiled shared-memory kernel:

```cpp
sharedABMultiply()
```

The function:

```cpp
ABMultiply()
```

configures and launches the tiled CUDA kernel.

---

## 12. Command-Line Arguments

The executable receives three arguments:

```text
mmb_nvshmem <device_id> <matrix_size> <communication_configuration>
```

where:

| Argument                      | Description                                 |
| ----------------------------- | ------------------------------------------- |
| `device_id`                   | CUDA device associated with the MPI process |
| `matrix_size`                 | Dimension of the square matrices            |
| `communication_configuration` | Three-character communication configuration |

For example:

```bash
./mmb_nvshmem 0 8192 SNS
```

selects:

```text
CUDA Device : 0
Matrix Size : 8192 x 8192

A : NVSHMEM
B : NCCL
C : NVSHMEM
```

The communication configuration must contain exactly three characters selected from:

```text
M
C
N
S
```

---

## 13. Example Communication Configurations

### MPI only

```text
MMM
```

```text
A -> MPI
B -> MPI
C -> MPI
```

### CUDA-Aware MPI only

```text
CCC
```

```text
A -> CUDA-Aware MPI
B -> CUDA-Aware MPI
C -> CUDA-Aware MPI
```

### NCCL only

```text
NNN
```

```text
A -> NCCL
B -> NCCL
C -> NCCL
```

### NVSHMEM only

```text
SSS
```

```text
A -> NVSHMEM
B -> NVSHMEM
C -> NVSHMEM
```

### Hybrid example

```text
SNS
```

```text
A -> NVSHMEM
B -> NCCL
C -> NVSHMEM
```

This approach allows communication strategies to be mixed independently for each matrix.

---

## 14. Performance Measurement

The execution time is measured using:

```cpp
MPI_Wtime()
```

The benchmark executes the multiplication multiple times:

```cpp
const int loop_count = 10;
```

The average execution time is calculated as:

```cpp
elapsed_time = stop_time - start_time;

double avg_time = elapsed_time / (double)loop_count;
```

This enables comparisons among configurations such as:

```text
MMM
CCC
NNN
SSS
SNS
SNC
NCS
MSN
```

for the same computational workload and matrix size.

---

## 14. Benchmarking Purpose

The main purpose of this code is to evaluate the impact of **communication mechanisms and data movement** on distributed multi-GPU matrix multiplication.

Although the computational operation remains the same:

$$
C_i = A_i \times B
$$

the way data are moved between CPUs, GPUs, and remote nodes can significantly affect application performance.

The benchmark can therefore be used to investigate:

* host-to-device data movement;
* GPU-to-GPU communication;
* inter-node communication;
* MPI communication overhead;
* CUDA-Aware MPI;
* NCCL collective communication;
* NVSHMEM one-sided communication;
* symmetric GPU memory;
* communication/computation balance;
* scalability across multiple GPUs;
* scalability across multiple compute nodes;
* effects of communication topology;
* data locality.

---

## 15. MPI vs CUDA-Aware MPI vs NCCL vs NVSHMEM

The benchmark provides a common computational workload for comparing four communication models:

```text
                    Communication Models

          +-----------------------------------+
          |                                   |
          |              MPI                  |
          |        Host-oriented model        |
          |                                   |
          +-----------------------------------+

          +-----------------------------------+
          |                                   |
          |        CUDA-Aware MPI             |
          |     Direct GPU-buffer support     |
          |                                   |
          +-----------------------------------+

          +-----------------------------------+
          |                                   |
          |              NCCL                 |
          |       GPU collective model        |
          |                                   |
          +-----------------------------------+

          +-----------------------------------+
          |                                   |
          |            NVSHMEM                |
          |    GPU symmetric-memory model     |
          |    One-sided communication        |
          |                                   |
          +-----------------------------------+
```

This provides a controlled environment for investigating how different programming and communication models influence multi-GPU application performance.

---

## 16. Research Context

Modern GPUs provide extremely high computational throughput.

As GPU performance increases, however, application performance increasingly depends on:

```text
Where are the data?
        +
How must the data move?
        +
How expensive is the communication path?
        =
Effective Application Performance
```

A high-performance GPU can remain idle or underutilized while waiting for the data required to perform its calculations.

Therefore:

> **GPU performance alone does not determine application performance. Data placement and data movement are increasingly important factors in modern HPC systems.**

The introduction of NVSHMEM makes this benchmark particularly useful for investigating communication models based on **Partitioned Global Address Space (PGAS)** and one-sided GPU communication.

The benchmark can therefore support experiments involving:

**Multi-GPU Communication · Data Movement · Data Locality · MPI · CUDA-Aware MPI · NCCL · NVSHMEM · PGAS · High-Performance Computing**

---

## 17. Data Locality Perspective

The communication configuration can also be interpreted from a **Data Locality** perspective.

The application contains several levels of data movement:

```text
Host Memory
     |
     | PCIe / GPU Direct
     v
Local GPU Memory
     |
     | NVLink / PCIe
     v
Remote GPU
     |
     | Network / InfiniBand
     v
GPU on Remote Node
```

Different communication libraries may use these paths differently.

The benchmark therefore provides a foundation for studying how:

* data placement;
* communication mechanism;
* GPU topology;
* network topology;
* remote memory access;

affect the execution time of a distributed multi-GPU application.

---
