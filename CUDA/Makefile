# Target executable name
TARGET = axpy

# CUDA source file
CUDA_FILE = BLAS/axpy.cu

# C++ source file
CPP_FILE = main.cpp

# Include directory
INCLUDE_DIR =-I.

# CUDA libraries path (adjust according to your conda installation)
CUDA_LIBS = $(shell echo ${CONDA_PREFIX}/lib64 | sed 's:/lib64::')/libnvrtc.so \
           $(shell echo ${CONDA_PREFIX}/lib64 | sed 's:/lib64::')/libcublas.so

# NVCC flags
NVCC_FLAGS = -O3 -arch=compute_86 -Xcompiler -Wall -ccbin=$(which nvcc)

# C++ compiler flags
CPP_FLAGS = -O3 -Wall

# Define paths based on conda environment
ifeq ($(shell which conda),)
  $(error Could not locate conda. Please ensure conda is installed and in your PATH.)
else
  CONDA_PREFIX := $(shell conda env list | grep '*' | awk '{print $NF}')
endif

.PHONY: clean all

all: $(TARGET)

$(TARGET): $(CPP_FILE) $(CUDA_FILE)
	nvcc $(NVCC_FLAGS) $(INCLUDE_DIR) $(CUDA_FILE) -o $(TARGET).o
	g++ $(CPP_FLAGS) $(INCLUDE_DIR) $(TARGET).o $(CUDA_LIBS) -o $(TARGET)
	rm -f $(TARGET).o

clean:
	rm -rf $(TARGET) $(TARGET).o

