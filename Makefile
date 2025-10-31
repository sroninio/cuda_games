# Makefile for CUDA prefix_sum program

# Compiler
NVCC = nvcc

# Compiler flags
NVCC_FLAGS = -arch=sm_50

# Target executable
TARGET = prefix_sum

# Source file
SOURCE = prefix_sum.cu

# Default target
all: $(TARGET)

# Compile the CUDA program
$(TARGET): $(SOURCE)
	$(NVCC) $(NVCC_FLAGS) $(SOURCE) -o $(TARGET)

# Run the program
run: $(TARGET)
	./$(TARGET)

# Clean up
clean:
	rm -f $(TARGET)

.PHONY: all run clean

