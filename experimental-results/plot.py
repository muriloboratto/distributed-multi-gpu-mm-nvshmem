#!/usr/bin/env python3

import re
import matplotlib.pyplot as plt

# ============================================================
# Input files
# ============================================================

files = {
    "MPI": "result--2048-32768-1node-4GPUs-MMM.txt",
    "CUDA-Aware MPI": "result--2048-32768-1node-4GPUs-CCC.txt",
    "NCCL": "result--2048-32768-1node-4GPUs-NNN.txt",
    "NVSHMEM": "result--2048-32768-1node-4GPUs-SSS.txt",
}

# ============================================================
# Read benchmark results
# ============================================================

def read_results(filename):
    results = {}

    pattern = re.compile(
        r"Matrix size:\s*(\d+).*?"
        r"Time \(seconds\):\s*([0-9.eE+-]+)"
    )

    with open(filename, "r") as file:
        for line in file:
            match = pattern.search(line)

            if match:
                matrix_size = int(match.group(1))
                execution_time = float(match.group(2))

                results[matrix_size] = execution_time

    return results


data = {}

for library, filename in files.items():
    data[library] = read_results(filename)

# ============================================================
# Select matrix sizes available for all libraries
# ============================================================

common_sizes = set(data["MPI"].keys())

for library in data:
    common_sizes &= set(data[library].keys())

matrix_sizes = sorted(common_sizes)

if not matrix_sizes:
    raise RuntimeError(
        "No common matrix sizes were found in all result files."
    )

# ============================================================
# Display extracted data
# ============================================================

print("\nExecution Time (seconds)\n")

print(
    f"{'Matrix':>10}"
    f"{'MPI':>12}"
    f"{'CUDA-Aware':>15}"
    f"{'NCCL':>12}"
    f"{'NVSHMEM':>12}"
)

for size in matrix_sizes:
    print(
        f"{size:>10}"
        f"{data['MPI'][size]:>12.3f}"
        f"{data['CUDA-Aware MPI'][size]:>15.3f}"
        f"{data['NCCL'][size]:>12.3f}"
        f"{data['NVSHMEM'][size]:>12.3f}"
    )

# ============================================================
# Plot 1 - Execution Time
# ============================================================

plt.figure(figsize=(9, 6))

for library in files:
    times = [data[library][size] for size in matrix_sizes]

    plt.plot(
        matrix_sizes,
        times,
        marker="o",
        linewidth=2,
        label=library
    )

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Execution Time (s)")
plt.title("Matrix Multiplication — Execution Time\n1 Node / 4 GPUs")

plt.xticks(matrix_sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "execution_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Calculate Speedup
# ============================================================

speedup = {}

for library in files:

    speedup[library] = []

    for size in matrix_sizes:

        mpi_time = data["MPI"][size]
        library_time = data[library][size]

        speedup_value = mpi_time / library_time

        speedup[library].append(speedup_value)

# ============================================================
# Display Speedup
# ============================================================

print("\nSpeedup relative to MPI\n")

print(
    f"{'Matrix':>10}"
    f"{'MPI':>12}"
    f"{'CUDA-Aware':>15}"
    f"{'NCCL':>12}"
    f"{'NVSHMEM':>12}"
)

for i, size in enumerate(matrix_sizes):

    print(
        f"{size:>10}"
        f"{speedup['MPI'][i]:>12.3f}"
        f"{speedup['CUDA-Aware MPI'][i]:>15.3f}"
        f"{speedup['NCCL'][i]:>12.3f}"
        f"{speedup['NVSHMEM'][i]:>12.3f}"
    )

# ============================================================
# Plot 2 - Speedup
# ============================================================

plt.figure(figsize=(9, 6))

for library in files:

    plt.plot(
        matrix_sizes,
        speedup[library],
        marker="o",
        linewidth=2,
        label=library
    )

# MPI baseline
plt.axhline(
    y=1.0,
    linestyle="--",
    linewidth=1
)

plt.xlabel("Matrix Size (N × N)")
plt.ylabel("Speedup")
plt.title(
    "Matrix Multiplication — Speedup Relative to MPI\n"
    "1 Node / 4 GPUs"
)

plt.xticks(matrix_sizes)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()

plt.savefig(
    "speedup.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# ============================================================
# Final information
# ============================================================

print("\nGenerated files:")
print("  execution_time.png")
print("  speedup.png")
