CC = mpic++
NVCC = nvcc

CUDA_DIR = $(CUDA_HOME)
CUDA_INC = -I$(CUDA_DIR)/include -I$(CUDA_DIR)/samples/common/inc
CUDA_LIB = -L$(CUDA_DIR)/lib64

NVSHMEM_DIR = $(NVSHMEM_HOME)
NVSHMEM_INC = -I$(NVSHMEM_DIR)/include
NVSHMEM_LIB = -L$(NVSHMEM_DIR)/lib

NCCL_DIR = $(NCCL_HOME)
NCCL_INC = -I$(NCCL_DIR)/include
NCCL_LIB = -L$(NCCL_DIR)/lib

CUDA_FLAGS = -O2 -rdc=true -Wno-deprecated-gpu-targets \
             -gencode=arch=compute_70,code=sm_70 \
             -Xcompiler -fopenmp

LINK_FLAGS = -rdc=true \
             -gencode=arch=compute_70,code=sm_70 \
             -Xcompiler -fopenmp

LD_LIBS = -lnvidia-ml -lcudart -lnvshmem -lcuda -lnccl -lm

EXE = mmb
OBJ_C = $(EXE).o

CUFILE = mulmat_kernel
OBJ_CUDA = $(CUFILE).o

$(EXE): $(OBJ_C) $(OBJ_CUDA)
	$(NVCC) $(LINK_FLAGS) \
	        -ccbin $(CC) \
	        $(OBJ_C) $(OBJ_CUDA) \
	        -o $(EXE) \
	        $(CUDA_LIB) \
	        $(NVSHMEM_LIB) \
	        $(NCCL_LIB) \
	        $(LD_LIBS)

$(OBJ_C): $(EXE).cu
	$(NVCC) $(CUDA_FLAGS) \
	        -ccbin $(CC) \
	        $(CUDA_INC) \
	        $(NVSHMEM_INC) \
	        $(NCCL_INC) \
	        -c $< \
	        -o $@

$(OBJ_CUDA): $(CUFILE).cu
	$(NVCC) $(CUDA_FLAGS) \
	        -ccbin $(CC) \
	        $(CUDA_INC) \
	        $(NVSHMEM_INC) \
	        $(NCCL_INC) \
	        -c $< \
	        -o $@

clean:
	rm -f *.o $(EXE)
